# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_mqa_logits,
    fp8_mqa_logits_torch,
    fp8_paged_mqa_logits,
    fp8_paged_mqa_logits_torch,
    is_deep_gemm_supported,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._xpu_ops import xpu_ops as ops

logger = init_logger(__name__)


def _cache_bf16_k(
    kv_cache: torch.Tensor,
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    flat_cache = kv_cache.view(-1, kv_cache.shape[-1])
    valid = slot_mapping >= 0
    if valid.any():
        flat_cache[slot_mapping[valid].long()] = k[valid]
    return flat_cache


def _gather_bf16_req_k(
    flat_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    positions = torch.arange(seq_len, device=flat_cache.device, dtype=torch.long)
    slots = block_table_row[positions // block_size].long() * block_size + (
        positions % block_size
    )
    return flat_cache[slots]


def _bf16_mqa_logits_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    seq_len_kv = k.shape[0]
    mask_lo = (
        torch.arange(0, seq_len_kv, device=q.device)[None, :] >= cu_seqlen_ks[:, None]
    )
    mask_hi = (
        torch.arange(0, seq_len_kv, device=q.device)[None, :] < cu_seqlen_ke[:, None]
    )
    mask = mask_lo & mask_hi

    k_float = k.float()
    score = torch.einsum("mhd,nd->hmn", q.float(), k_float) * softmax_scale
    logits = (score * weights.unsqueeze(-1).transpose(0, 1).float()).sum(dim=0)
    return logits.masked_fill(~mask, float("-inf"))


def _bf16_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
    softmax_scale: float,
) -> torch.Tensor:
    batch_size, next_n, _, _ = q.size()
    _, block_size, dim = kv_cache.size()
    q = q.float()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    for i in range(batch_size):
        context_len = int(context_lens[i].item())
        if context_len <= 0:
            continue
        q_offsets = torch.arange(context_len - next_n, context_len, device=q.device)
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :]
            .transpose(0, 1)
            .contiguous()
            .float()
        )
        for block_idx in range(cdiv(context_len, block_size)):
            block_start = block_idx * block_size
            block_end = min(block_start + block_size, max_model_len)
            if block_end <= block_start:
                continue
            block_len = block_end - block_start
            block_id = block_tables[i][block_idx]
            qx = q[i]
            kx = kv_cache[block_id, :block_len].float()
            k_offsets = torch.arange(block_start, block_end, device=q.device)
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            score = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1)).to(logits.dtype)
                * softmax_scale,
                float("-inf"),
            )
            logits_block = score * weight_slice[..., None]
            logits_block = logits_block.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_start:block_end,
            ] = torch.where(
                k_offsets[None, :] <= q_offsets[:, None], logits_block, float("-inf")
            )
    return logits


def _gather_bf16_chunk_k(
    flat_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    total_seq_lens = int(cu_seq_lens[-1].item())
    gathered = torch.empty(
        total_seq_lens,
        flat_cache.shape[-1],
        dtype=flat_cache.dtype,
        device=flat_cache.device,
    )
    for req_idx in range(block_table.shape[0]):
        start = int(cu_seq_lens[req_idx].item())
        end = int(cu_seq_lens[req_idx + 1].item())
        seq_len = end - start
        if seq_len <= 0:
            continue
        gathered[start:end] = _gather_bf16_req_k(
            flat_cache,
            block_table[req_idx],
            seq_len,
            block_size,
        )
    return gathered


def sparse_attn_indexer_bf16(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    topk_indices_buffer: torch.Tensor,
    softmax_scale: float,
    max_total_seq_len: int | None = None,
) -> torch.Tensor:
    # GLM5 keeps the bf16/fp32 score math from the HF reference, but should
    # otherwise follow the same prefill/decode traversal and top-k flow as the
    # stock vLLM indexer.
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        if max_total_seq_len is not None:
            current_workspace_manager().get_simultaneous(
                ((max_total_seq_len, k.shape[-1]), kv_cache.dtype),
            )
        return topk_indices_buffer

    attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata.slot_mapping
    num_tokens = slot_mapping.shape[0]
    k = k[:num_tokens]
    q = q[:num_tokens]
    weights = weights[:num_tokens]

    flat_cache = _cache_bf16_k(kv_cache, k, slot_mapping)
    block_size = kv_cache.shape[1]
    topk_indices_buffer[: hidden_states.shape[0]] = -1

    if attn_metadata.num_prefills > 0:
        assert attn_metadata.prefill is not None
        workspace_manager = current_workspace_manager()
        max_chunk_tokens = max(
            chunk.total_seq_lens for chunk in attn_metadata.prefill.chunks
        )
        [gathered_k_full] = workspace_manager.get_simultaneous(
            ((max_chunk_tokens, k.shape[-1]), kv_cache.dtype),
        )
        for chunk in attn_metadata.prefill.chunks:
            gathered_k = gathered_k_full[: chunk.total_seq_lens]
            if current_platform.is_cuda():
                ops.cp_gather_cache(
                    src_cache=kv_cache,
                    dst=gathered_k,
                    block_table=chunk.block_table,
                    cu_seq_lens=chunk.cu_seq_lens,
                    batch_size=chunk.num_reqs,
                    seq_starts=None,
                )
            else:
                gathered_k.copy_(
                    _gather_bf16_chunk_k(
                        flat_cache,
                        chunk.block_table,
                        chunk.cu_seq_lens,
                        block_size,
                    )
                )
            logits = _bf16_mqa_logits_torch(
                q[chunk.token_start : chunk.token_end],
                gathered_k,
                weights[chunk.token_start : chunk.token_end],
                chunk.cu_seqlen_ks,
                chunk.cu_seqlen_ke,
                softmax_scale,
            )
            num_rows = logits.shape[0]
            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]
            if current_platform.is_xpu():
                ops.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            else:
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

    if attn_metadata.num_decodes > 0:
        assert attn_metadata.decode is not None
        decode = attn_metadata.decode
        decode_lens = decode.decode_lens
        if decode.requires_padding:
            padded_q = pack_seq_triton(
                q[: attn_metadata.num_decode_tokens], decode_lens
            )
        else:
            padded_q = q[: attn_metadata.num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q.shape[1:]
            )
        batch_size = padded_q.shape[0]
        next_n = padded_q.shape[1]
        num_padded_tokens = batch_size * next_n
        logits = _bf16_paged_mqa_logits_torch(
            padded_q,
            kv_cache,
            weights[:num_padded_tokens],
            decode.seq_lens,
            decode.block_table,
            max_model_len=attn_metadata.max_seq_len,
            softmax_scale=softmax_scale,
        )
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]
        num_rows = logits.shape[0]
        if decode.use_large_context_topk:
            if next_n == 1:
                lengths = decode.seq_lens
            else:
                lengths = (
                    decode.seq_lens.unsqueeze(1) - next_n + 1 + decode.offsets
                ).flatten()
            torch.ops._C.large_context_topk(logits, topk_indices, lengths, None)
        else:
            if current_platform.is_xpu():
                ops.top_k_per_row_decode(
                    logits,
                    next_n,
                    decode.seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            else:
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    decode.seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
        if decode.requires_padding:
            unpacked = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[
                : attn_metadata.num_decode_tokens, : unpacked.shape[-1]
            ] = unpacked

    return topk_indices_buffer


def sparse_attn_indexer_bf16_op(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_total_seq_len: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    return sparse_attn_indexer_bf16(
        hidden_states,
        k_cache_prefix,
        kv_cache,
        q,
        k,
        weights,
        topk_tokens,
        topk_indices_buffer,
        head_dim**-0.5,
        max_total_seq_len=max_total_seq_len,
    )


def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        current_workspace_manager().get_simultaneous(
            ((total_seq_lens, head_dim), torch.float8_e4m3fn),
            ((total_seq_lens, 4), torch.uint8),
        )
        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
        )
    attn_metadata = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata.slot_mapping
    has_decode = attn_metadata.num_decodes > 0
    has_prefill = attn_metadata.num_prefills > 0
    num_decode_tokens = attn_metadata.num_decode_tokens

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    k = k[:num_tokens]

    ops.indexer_k_quant_and_cache(
        k,
        kv_cache,
        slot_mapping,
        quant_block_size,
        scale_fmt,
    )

    topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata.prefill

        # Get the full shared workspace buffers once (will allocate on first use)
        workspace_manager = current_workspace_manager()
        k_fp8_full, k_scale_full = workspace_manager.get_simultaneous(
            ((total_seq_lens, head_dim), fp8_dtype),
            ((total_seq_lens, 4), torch.uint8),
        )
        for chunk in prefill_metadata.chunks:
            k_fp8 = k_fp8_full[: chunk.total_seq_lens]
            k_scale = k_scale_full[: chunk.total_seq_lens]
            ops.cp_gather_indexer_k_quant_cache(
                kv_cache,
                k_fp8,
                k_scale,
                chunk.block_table,
                chunk.cu_seq_lens,
            )
            if is_deep_gemm_supported():
                logits = fp8_mqa_logits(
                    q_fp8[chunk.token_start : chunk.token_end],
                    (k_fp8, k_scale.view(torch.float32).flatten()),
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    clean_logits=False,
                )
            else:
                logits = fp8_mqa_logits_torch(
                    q_fp8[chunk.token_start : chunk.token_end],
                    (k_fp8, k_scale.view(torch.float32).flatten()),
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                )
            num_rows = logits.shape[0]

            topk_indices = topk_indices_buffer[
                chunk.token_start : chunk.token_end, :topk_tokens
            ]

            if current_platform.is_xpu():
                ops.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            else:
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

            # Compute lengths from row spans
            # lengths = (chunk.cu_seqlen_ke - chunk.cu_seqlen_ks).to(torch.int32)
            # torch.ops._C.large_context_topk(
            #    logits,
            #    topk_indices,
            #    lengths,
            #    chunk.cu_seqlen_ks,  # row_starts
            # )

    if has_decode:
        decode_metadata = attn_metadata.decode
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:]
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        assert batch_size == decode_metadata.seq_lens.shape[0]
        num_padded_tokens = batch_size * next_n
        if is_deep_gemm_supported():
            logits = fp8_paged_mqa_logits(
                padded_q_fp8_decode_tokens,
                kv_cache,
                weights[:num_padded_tokens],
                decode_metadata.seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                clean_logits=False,
            )
        else:
            logits = fp8_paged_mqa_logits_torch(
                padded_q_fp8_decode_tokens,
                kv_cache,
                weights[:num_padded_tokens],
                decode_metadata.seq_lens,
                decode_metadata.block_table,
                max_model_len=max_model_len,
            )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if decode_metadata.use_large_context_topk:
            if next_n == 1:
                lengths = decode_metadata.seq_lens
            else:
                # (bs,) -> (bs, 1) + (next_n,) -> (bs, next_n) -> (bs * next_n,)
                lengths = (
                    decode_metadata.seq_lens.unsqueeze(1)
                    - next_n
                    + 1
                    + decode_metadata.offsets
                ).flatten()

            torch.ops._C.large_context_topk(
                logits,
                topk_indices,
                lengths,
                None,
            )
        else:
            if current_platform.is_xpu():
                ops.top_k_per_row_decode(
                    logits,
                    next_n,
                    decode_metadata.seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            else:
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    decode_metadata.seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )

        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[:num_decode_tokens, : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    return topk_indices_buffer


def sparse_attn_indexer_bf16_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk_tokens: int,
    head_dim: int,
    max_total_seq_len: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)

direct_register_custom_op(
    op_name="sparse_attn_indexer_bf16",
    op_func=sparse_attn_indexer_bf16_op,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_bf16_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
        use_bf16_scoring: bool = False,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        self.use_bf16_scoring = use_bf16_scoring
        if (
            not self.use_bf16_scoring
            and current_platform.is_cuda()
            and not is_deep_gemm_supported()
        ):
            logger.warning_once(
                "DeepGEMM is not supported or available. SparseAttnIndexer will use a "
                "less efficient PyTorch implementation. "
                "Please make sure you have the required hardware and software setup "
                "for DeepGEMM to achieve optimal performance."
            )

    def _forward_bf16_scoring(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return torch.ops.vllm.sparse_attn_indexer_bf16(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache[0],
            q,
            k,
            weights,
            self.topk_tokens,
            self.head_dim,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if self.use_bf16_scoring:
            return self._forward_bf16_scoring(hidden_states, q_fp8, k, weights)
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_fp8, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_fp8, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if self.use_bf16_scoring:
            return self._forward_bf16_scoring(hidden_states, q_fp8, k, weights)
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache[0],
            q_fp8,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if self.use_bf16_scoring:
            return self._forward_bf16_scoring(hidden_states, q_fp8, k, weights)
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                self.k_cache.prefix,
                self.k_cache.kv_cache[0],
                q_fp8,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )
        else:
            raise RuntimeError(
                "Sparse attention indexer ROCm custom op requires ROCm "
                "Aiter ops to be enabled."
            )
