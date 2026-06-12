# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark GLM-5.1 MoE LoRA through FlashInfer CUTLASS vs Triton.

The default shape matches the GLM-5.1 MoE layers used by the local adapter:
M=32768, hidden=6144, moe_intermediate=2048, experts=256, top_k=8, rank=16.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

from flashinfer.autotuner import AutoTuner, autotune  # noqa: E402
from flashinfer.fused_moe import cutlass_fused_moe  # noqa: E402
from flashinfer.fused_moe.core import (  # noqa: E402
    ActivationType,
    get_cutlass_fused_moe_module,
)
from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU  # noqa: E402
from vllm.model_executor.layers.fused_moe.activation import (  # noqa: E402
    MoEActivation,
)
from vllm.model_executor.layers.fused_moe.config import (  # noqa: E402
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.lora_context import (  # noqa: E402
    MoELoRAContext,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import (  # noqa: E402
    TritonExperts,
)


DEFAULT_ADAPTER_PATH = Path(
    "/mnt/data/user/songlin/sft_training/checkpoints/glm51-sft/"
    "glm51-megatron-lora/global_step_124/huggingface/adapter"
)
DEFAULT_REQUESTED_BASE_PATH = Path("/mnt/data/user/songlin/kernel/flashinfer")

HIDDEN_SIZE = 6144
INTERMEDIATE_SIZE = 2048
NUM_EXPERTS = 256
TOP_K = 8
RANK = 16
DTYPE = torch.bfloat16


def _adapter_config(adapter_path: Path) -> dict:
    with (adapter_path / "adapter_config.json").open() as f:
        return json.load(f)


def _resolve_base_model_path(requested_path: Path, adapter_path: Path) -> Path:
    if (requested_path / "model.safetensors.index.json").exists():
        return requested_path
    adapter_base = _adapter_config(adapter_path).get("base_model_name_or_path")
    if adapter_base:
        adapter_base_path = Path(adapter_base)
        if (adapter_base_path / "model.safetensors.index.json").exists():
            return adapter_base_path
    raise FileNotFoundError(
        f"No model.safetensors.index.json under requested base path {requested_path} "
        f"or adapter base_model_name_or_path from {adapter_path}."
    )


def _make_balanced_routing(
    num_tokens: int,
    *,
    max_loras: int,
    include_no_lora: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)
    offsets = torch.arange(TOP_K, dtype=torch.int32, device=device) * (
        NUM_EXPERTS // TOP_K
    )
    topk_ids = (token_ids[:, None] + offsets[None, :]) % NUM_EXPERTS
    weights = torch.rand((num_tokens, TOP_K), dtype=torch.float32, device=device)
    topk_weights = weights / weights.sum(dim=1, keepdim=True)
    if include_no_lora:
        lora_cycle = max_loras + 1
        token_lora_indices = token_ids % lora_cycle
        token_lora_indices = torch.where(
            token_lora_indices == max_loras,
            torch.full_like(token_lora_indices, -1),
            token_lora_indices,
        )
    else:
        token_lora_indices = token_ids % max_loras
    return topk_ids.contiguous(), topk_weights.contiguous(), token_lora_indices


def _make_lora_ptrs(lora_a: torch.Tensor, lora_b: torch.Tensor) -> torch.Tensor:
    assert lora_a.is_contiguous() and lora_b.is_contiguous()
    a_step = lora_a.stride(0) * lora_a.element_size()
    b_step = lora_b.stride(0) * lora_b.element_size()
    ptrs: list[int] = []
    for lora_id in range(lora_a.shape[0]):
        ptrs.append(lora_a.data_ptr() + lora_id * a_step)
        ptrs.append(lora_b.data_ptr() + lora_id * b_step)
    return torch.tensor(ptrs, dtype=torch.int64, device=lora_a.device)


def _load_adapter_lora(
    adapter_path: Path,
    *,
    layer: int,
    max_loras: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    config = _adapter_config(adapter_path)
    scale = float(config.get("lora_alpha", RANK)) / float(config.get("r", RANK))
    weights_path = adapter_path / "adapter_model.safetensors"

    gate_a = torch.empty((NUM_EXPERTS, RANK, HIDDEN_SIZE), dtype=DTYPE, device=device)
    gate_b = torch.empty((NUM_EXPERTS, INTERMEDIATE_SIZE, RANK), dtype=DTYPE, device=device)
    up_a = torch.empty_like(gate_a)
    up_b = torch.empty_like(gate_b)
    down_a = torch.empty((NUM_EXPERTS, RANK, INTERMEDIATE_SIZE), dtype=DTYPE, device=device)
    down_b = torch.empty((NUM_EXPERTS, HIDDEN_SIZE, RANK), dtype=DTYPE, device=device)

    prefix = f"base_model.model.model.layers.{layer}.mlp.experts"
    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        for expert in range(NUM_EXPERTS):
            base = f"{prefix}.{expert}"
            gate_a[expert].copy_(
                f.get_tensor(f"{base}.gate_proj.lora_A.weight").to(device=device)
            )
            gate_b[expert].copy_(
                (f.get_tensor(f"{base}.gate_proj.lora_B.weight") * scale).to(
                    device=device
                )
            )
            up_a[expert].copy_(
                f.get_tensor(f"{base}.up_proj.lora_A.weight").to(device=device)
            )
            up_b[expert].copy_(
                (f.get_tensor(f"{base}.up_proj.lora_B.weight") * scale).to(
                    device=device
                )
            )
            down_a[expert].copy_(
                f.get_tensor(f"{base}.down_proj.lora_A.weight").to(device=device)
            )
            down_b[expert].copy_(
                (f.get_tensor(f"{base}.down_proj.lora_B.weight") * scale).to(
                    device=device
                )
            )

    shared_gate_up_a = torch.equal(gate_a, up_a)

    def replicate(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0).expand(max_loras, *t.shape).contiguous()

    gate_a_stacked = replicate(gate_a)
    up_a_stacked = gate_a_stacked if shared_gate_up_a else replicate(up_a)

    return {
        "gate_a": gate_a_stacked,
        "gate_b": replicate(gate_b),
        "up_a": up_a_stacked,
        "up_b": replicate(up_b),
        "down_a": replicate(down_a),
        "down_b": replicate(down_b),
    }


def _load_base_moe_weights(
    base_model_path: Path,
    *,
    layer: int,
    device: torch.device,
    random_weights: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    w1 = torch.empty(
        (NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE),
        dtype=DTYPE,
        device=device,
    )
    w2 = torch.empty(
        (NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE),
        dtype=DTYPE,
        device=device,
    )
    if random_weights:
        w1.normal_(mean=0.0, std=1.0 / math.sqrt(HIDDEN_SIZE))
        w2.normal_(mean=0.0, std=1.0 / math.sqrt(INTERMEDIATE_SIZE))
        return w1, w2

    index_path = base_model_path / "model.safetensors.index.json"
    with index_path.open() as f:
        weight_map = json.load(f)["weight_map"]

    keys_by_file: dict[str, list[tuple[int, str, str]]] = {}
    for expert in range(NUM_EXPERTS):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"
            keys_by_file.setdefault(weight_map[key], []).append((expert, proj, key))

    for filename, entries in sorted(keys_by_file.items()):
        with safe_open(str(base_model_path / filename), framework="pt", device="cpu") as f:
            for expert, proj, key in entries:
                src = f.get_tensor(key).to(device=device)
                if proj == "gate_proj":
                    w1[expert, :INTERMEDIATE_SIZE].copy_(src)
                elif proj == "up_proj":
                    w1[expert, INTERMEDIATE_SIZE:].copy_(src)
                else:
                    w2[expert].copy_(src)
    return w1.contiguous(), w2.contiguous()


def _make_triton_experts(device: torch.device, max_tokens: int) -> TritonExperts:
    moe_config = FusedMoEConfig(
        num_experts=NUM_EXPERTS,
        experts_per_token=TOP_K,
        hidden_dim=HIDDEN_SIZE,
        hidden_dim_unpadded=HIDDEN_SIZE,
        intermediate_size_per_partition=INTERMEDIATE_SIZE,
        intermediate_size_per_partition_unpadded=INTERMEDIATE_SIZE,
        num_local_experts=NUM_EXPERTS,
        num_logical_experts=NUM_EXPERTS,
        activation=MoEActivation.SILU,
        device=device,
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        in_dtype=DTYPE,
        max_num_tokens=max_tokens,
        is_lora_enabled=True,
    )
    return TritonExperts(moe_config, FUSED_MOE_UNQUANTIZED_CONFIG)


def _make_lora_context(
    lora: dict[str, torch.Tensor],
    token_lora_indices: torch.Tensor,
    *,
    max_loras: int,
    device: torch.device,
    max_tokens: int,
) -> MoELoRAContext:
    lora_config = SimpleNamespace(max_loras=max_loras, specialize_active_lora=False)
    punica = PunicaWrapperGPU(
        max_num_batched_tokens=max_tokens,
        max_batches=max_tokens,
        device=device,
        lora_config=lora_config,
    )
    punica.token_mapping_meta.prepare_tensors(token_lora_indices)
    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32, device=device)
    return MoELoRAContext(
        w13_lora_a_stacked=(lora["gate_a"], lora["up_a"]),
        w13_lora_b_stacked=(lora["gate_b"], lora["up_b"]),
        w2_lora_a_stacked=(lora["down_a"],),
        w2_lora_b_stacked=(lora["down_b"],),
        adapter_enabled=adapter_enabled,
        max_loras=max_loras,
        top_k=TOP_K,
        w13_num_slices=2,
        fully_sharded=False,
        tp_rank=0,
        tp_size=1,
        local_num_experts=NUM_EXPERTS,
        punica_wrapper=punica,
        use_tuned_config=False,
        local_token_lora_mapping=token_lora_indices,
    )


def _time_cuda(fn, *, warmups: int, repeats: int) -> tuple[float, float, float]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop))
    times.sort()
    mid = times[len(times) // 2]
    return mid, min(times), max(times)


def _read_flashinfer_autotuned_profile_ids(cache_path: Path) -> tuple[int, int] | None:
    if not cache_path.exists():
        return None
    with cache_path.open() as f:
        configs = json.load(f)
    gemm1 = None
    gemm2 = None
    for key, value in configs.items():
        if key == "_metadata":
            continue
        if not isinstance(value, list) or len(value) != 2:
            continue
        tactic = int(value[1])
        if key.startswith("('trtllm::fused_moe::gemm1',"):
            gemm1 = tactic
        elif key.startswith("('trtllm::fused_moe::gemm2',"):
            gemm2 = tactic
    if gemm1 is None or gemm2 is None:
        return None
    return gemm1, gemm2


def _parse_profile_sweep(value: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for raw_pair in value.replace(":", ";").split(";"):
        raw_pair = raw_pair.strip()
        if not raw_pair:
            continue
        parts = [part.strip() for part in raw_pair.split(",")]
        if len(parts) != 2:
            raise ValueError(
                "--flashinfer-profile-sweep expects pairs like '22,85;23,86'."
            )
        pairs.append((int(parts[0]), int(parts[1])))
    if not pairs:
        raise ValueError("--flashinfer-profile-sweep did not contain any profile pairs.")
    return pairs


def _print_flashinfer_tactics(
    *,
    topk_ids: torch.Tensor,
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> None:
    module = get_cutlass_fused_moe_module("90")
    closure = dict(
        zip(
            module.cutlass_fused_moe.__code__.co_freevars,
            module.cutlass_fused_moe.__closure__ or (),
        )
    )
    moe_runner_cls = closure["MoERunner"].cell_contents
    runner = moe_runner_cls(
        x_dtype=hidden.dtype,
        weight_dtype=w1.dtype,
        output_dtype=hidden.dtype,
        top_k=topk_ids.size(1),
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        cluster_size=1,
        cluster_rank=0,
        enable_alltoall=False,
        use_deepseek_fp8_block_scale=False,
        use_w4_group_scaling=False,
        use_mxfp8_act_scaling=False,
        min_latency_mode=False,
        enable_pdl=False,
        activation_type=ActivationType.Swiglu,
        use_packed_weights=False,
        use_lora=True,
    )
    fused_runner = runner.fused_moe_runner
    gemm1_count = int(fused_runner.get_gemm1_tactic_count())
    gemm2_count = int(fused_runner.get_gemm2_tactic_count())
    total_count = int(fused_runner.get_tactic_num())
    print(f"flashinfer_tactic_count={total_count}")
    print(f"flashinfer_gemm1_tactic_count={gemm1_count}")
    print(f"flashinfer_gemm2_tactic_count={gemm2_count}")

    valid_gemm1 = []
    valid_gemm2 = []
    finalize_gemm2 = []
    for tactic in range(total_count):
        try:
            occupancy = int(fused_runner.get_tactic_occupancy(tactic))
        except Exception as exc:
            print(f"tactic={tactic} occupancy_error={exc}")
            continue
        if occupancy <= 0:
            continue
        if tactic < gemm1_count:
            valid_gemm1.append(tactic)
        else:
            valid_gemm2.append(tactic)
            if bool(fused_runner.is_tactic_finalize_fusion(tactic)):
                finalize_gemm2.append(tactic)

    print("flashinfer_valid_gemm1_tactics=" + ",".join(map(str, valid_gemm1)))
    print("flashinfer_valid_gemm2_tactics=" + ",".join(map(str, valid_gemm2)))
    print(
        "flashinfer_finalize_fusion_gemm2_tactics="
        + ",".join(map(str, finalize_gemm2))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--max-loras", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--base-model-path", type=Path, default=DEFAULT_REQUESTED_BASE_PATH)
    parser.add_argument("--random-base-weights", action="store_true")
    parser.add_argument("--check-output", action="store_true")
    parser.add_argument(
        "--only",
        choices=("both", "triton", "flashinfer"),
        default="both",
        help="Limit timing to one provider during tuning sweeps.",
    )
    parser.add_argument(
        "--lora-pattern",
        choices=("include-no-lora", "all-lora"),
        default="include-no-lora",
        help="Token-to-adapter mapping used for the synthetic 32K routing pattern.",
    )
    parser.add_argument(
        "--flashinfer-profile-ids",
        type=str,
        default=None,
        help="Override FlashInfer static profile IDs, for example '22,86'.",
    )
    parser.add_argument(
        "--flashinfer-profile-sweep",
        type=str,
        default=None,
        help="Time semicolon-separated FlashInfer profile pairs, for example '22,85;23,86'.",
    )
    parser.add_argument(
        "--flashinfer-gemm2-sms",
        type=str,
        default=None,
        help="Override FLASHINFER_CUTLASS_MOE_TMA_GEMM2_SMS.",
    )
    parser.add_argument(
        "--flashinfer-pingpong",
        type=str,
        default=None,
        help="Override FLASHINFER_CUTLASS_MOE_SM90_BF16_PINGPONG.",
    )
    parser.add_argument(
        "--torch-profile",
        action="store_true",
        help="Print CUDA kernel attribution for one timed provider iteration.",
    )
    parser.add_argument(
        "--profile-provider",
        choices=("flashinfer", "triton"),
        default="flashinfer",
        help="Provider to profile when --torch-profile is set.",
    )
    parser.add_argument(
        "--torch-profile-trace",
        type=Path,
        default=None,
        help="Export a Chrome trace for --torch-profile.",
    )
    parser.add_argument(
        "--flashinfer-autotune-cache",
        type=Path,
        default=None,
        help="Run FlashInfer's built-in GEMM tactic tuner and save/read this cache.",
    )
    parser.add_argument(
        "--flashinfer-autotune-warmup",
        type=int,
        default=None,
        help="Override AutoTuner warmup iterations for --flashinfer-autotune-cache.",
    )
    parser.add_argument(
        "--flashinfer-autotune-repeat",
        type=int,
        default=None,
        help="Override AutoTuner repeat iterations for --flashinfer-autotune-cache.",
    )
    parser.add_argument(
        "--print-flashinfer-tactics",
        action="store_true",
        help="Print FlashInfer GEMM tactic counts and occupancy-valid tactic IDs.",
    )
    args = parser.parse_args()

    if args.flashinfer_profile_ids is not None:
        os.environ["FLASHINFER_CUTLASS_MOE_STATIC_PROFILE_IDS"] = (
            args.flashinfer_profile_ids
        )
    if args.flashinfer_gemm2_sms is not None:
        os.environ["FLASHINFER_CUTLASS_MOE_TMA_GEMM2_SMS"] = args.flashinfer_gemm2_sms
    if args.flashinfer_pingpong is not None:
        os.environ["FLASHINFER_CUTLASS_MOE_SM90_BF16_PINGPONG"] = (
            args.flashinfer_pingpong
        )

    torch.manual_seed(0)
    torch.cuda.set_device(0)
    device = torch.device("cuda")

    resolved_base_path = _resolve_base_model_path(args.base_model_path, args.adapter_path)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"tokens={args.tokens} top_k={TOP_K} experts={NUM_EXPERTS}")
    print(f"max_loras={args.max_loras} lora_pattern={args.lora_pattern}")
    print(f"adapter_path={args.adapter_path}")
    print(f"requested_base_model_path={args.base_model_path}")
    print(f"resolved_base_model_path={resolved_base_path}")
    print(f"base_weights={'random' if args.random_base_weights else 'safetensors'}")
    print(
        "flashinfer_profile_ids="
        f"{os.environ.get('FLASHINFER_CUTLASS_MOE_STATIC_PROFILE_IDS', 'default')}"
    )
    print(
        "flashinfer_gemm2_sms="
        f"{os.environ.get('FLASHINFER_CUTLASS_MOE_TMA_GEMM2_SMS', 'default')}"
    )
    print(
        "flashinfer_pingpong="
        f"{os.environ.get('FLASHINFER_CUTLASS_MOE_SM90_BF16_PINGPONG', 'default')}"
    )

    topk_ids, topk_weights, token_lora_indices = _make_balanced_routing(
        args.tokens,
        max_loras=args.max_loras,
        include_no_lora=args.lora_pattern == "include-no-lora",
        device=device,
    )
    hidden = torch.randn(
        (args.tokens, HIDDEN_SIZE),
        dtype=DTYPE,
        device=device,
    ) / math.sqrt(HIDDEN_SIZE)

    print("loading LoRA adapter weights...")
    lora = _load_adapter_lora(
        args.adapter_path,
        layer=args.layer,
        max_loras=args.max_loras,
        device=device,
    )
    print("loading base MoE weights...")
    w1, w2 = _load_base_moe_weights(
        resolved_base_path,
        layer=args.layer,
        device=device,
        random_weights=args.random_base_weights,
    )
    torch.cuda.synchronize()

    if args.print_flashinfer_tactics:
        _print_flashinfer_tactics(topk_ids=topk_ids, hidden=hidden, w1=w1, w2=w2)
        return

    triton_experts = None
    triton_workspace13 = None
    triton_workspace2 = None
    triton_out = torch.empty((args.tokens, HIDDEN_SIZE), dtype=DTYPE, device=device)
    if args.only != "flashinfer":
        triton_experts = _make_triton_experts(device, args.tokens)
        triton_experts.set_lora_context(
            _make_lora_context(
                lora,
                token_lora_indices,
                max_loras=args.max_loras,
                device=device,
                max_tokens=args.tokens,
            )
        )
        workspace13_shape, workspace2_shape, _ = triton_experts.workspace_shapes(
            args.tokens,
            2 * INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            TOP_K,
            NUM_EXPERTS,
            NUM_EXPERTS,
            None,
            MoEActivation.SILU,
        )
        triton_workspace13 = torch.empty(workspace13_shape, dtype=DTYPE, device=device)
        triton_workspace2 = torch.empty(workspace2_shape, dtype=DTYPE, device=device)
    flashinfer_out = torch.empty_like(triton_out)

    ranks = torch.full((args.max_loras,), RANK, dtype=torch.int32, device=device)
    gate_ptrs = _make_lora_ptrs(lora["gate_a"], lora["gate_b"])
    up_ptrs = _make_lora_ptrs(lora["up_a"], lora["up_b"])
    down_ptrs = _make_lora_ptrs(lora["down_a"], lora["down_b"])
    def run_triton() -> None:
        assert triton_experts is not None
        assert triton_workspace13 is not None
        assert triton_workspace2 is not None
        triton_experts.apply(
            triton_out,
            hidden,
            w1,
            w2,
            topk_weights,
            topk_ids,
            MoEActivation.SILU,
            NUM_EXPERTS,
            None,
            None,
            None,
            triton_workspace13,
            triton_workspace2,
            None,
            False,
        )

    def run_flashinfer() -> None:
        cutlass_fused_moe(
            input=hidden,
            token_selected_experts=topk_ids,
            token_final_scales=topk_weights,
            fc1_expert_weights=w1,
            fc2_expert_weights=w2,
            output_dtype=DTYPE,
            quant_scales=[],
            output=flashinfer_out,
            activation_type=ActivationType.Swiglu,
            tune_max_num_tokens=max(args.tokens, 8192),
            token_lora_indices=token_lora_indices,
            fc1_lora_ranks=ranks,
            fc1_lora_weight_ptrs=up_ptrs,
            fc2_lora_ranks=ranks,
            fc2_lora_weight_ptrs=down_ptrs,
            gated_lora_ranks=ranks,
            gated_lora_weight_ptrs=gate_ptrs,
            lora_max_rank=RANK,
        )

    if args.flashinfer_autotune_cache is not None and args.only != "triton":
        tuner = AutoTuner.get()
        if args.flashinfer_autotune_warmup is not None:
            tuner.warmup = args.flashinfer_autotune_warmup
        if args.flashinfer_autotune_repeat is not None:
            tuner.repeat = args.flashinfer_autotune_repeat
        print("autotuning FlashInfer GEMM tactics...")
        with autotune(
            True,
            cache=str(args.flashinfer_autotune_cache),
            tuning_buckets=(args.tokens,),
            round_up=True,
        ):
            run_flashinfer()
        torch.cuda.synchronize()
        tuned_profile_ids = _read_flashinfer_autotuned_profile_ids(
            args.flashinfer_autotune_cache
        )
        if tuned_profile_ids is None:
            print("flashinfer_autotuned_profile_ids=unavailable")
        else:
            tuned_profile_ids_text = f"{tuned_profile_ids[0]},{tuned_profile_ids[1]}"
            print(f"flashinfer_autotuned_profile_ids={tuned_profile_ids_text}")
            os.environ["FLASHINFER_CUTLASS_MOE_STATIC_PROFILE_IDS"] = (
                tuned_profile_ids_text
            )

    if args.flashinfer_profile_sweep is not None:
        if args.only == "triton":
            raise ValueError("--flashinfer-profile-sweep requires FlashInfer to be initialized")
        print("sweeping FlashInfer profile pairs...")
        for gemm1_id, gemm2_id in _parse_profile_sweep(args.flashinfer_profile_sweep):
            profile_ids_text = f"{gemm1_id},{gemm2_id}"
            os.environ["FLASHINFER_CUTLASS_MOE_STATIC_PROFILE_IDS"] = profile_ids_text
            torch.cuda.synchronize()
            flashinfer_ms, flashinfer_min, flashinfer_max = _time_cuda(
                run_flashinfer,
                warmups=args.warmups,
                repeats=args.repeats,
            )
            print(
                "profile_pair="
                f"{profile_ids_text} flashinfer_cutlass_ms={flashinfer_ms:.3f} "
                f"min={flashinfer_min:.3f} max={flashinfer_max:.3f}"
            )
        return

    if args.torch_profile:
        if args.profile_provider == "triton" and args.only == "flashinfer":
            raise ValueError("--profile-provider triton requires Triton to be initialized")
        if args.profile_provider == "flashinfer" and args.only == "triton":
            raise ValueError(
                "--profile-provider flashinfer requires FlashInfer to be initialized"
            )
        profile_fn = run_triton if args.profile_provider == "triton" else run_flashinfer
        for _ in range(args.warmups):
            profile_fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            profile_fn()
            torch.cuda.synchronize()
        if args.torch_profile_trace is not None:
            prof.export_chrome_trace(str(args.torch_profile_trace))
        print(
            prof.key_averages().table(
                sort_by="cuda_time_total",
                row_limit=40,
            )
        )
        return

    triton_ms = None
    flashinfer_ms = None
    if args.only != "flashinfer":
        print("warming and timing Triton baseline...")
        triton_ms, triton_min, triton_max = _time_cuda(
            run_triton,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        print(
            "triton_ms="
            f"{triton_ms:.3f} min={triton_min:.3f} max={triton_max:.3f}"
        )
    if args.only != "triton":
        print("warming and timing FlashInfer CUTLASS...")
        flashinfer_ms, flashinfer_min, flashinfer_max = _time_cuda(
            run_flashinfer,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        print(
            "flashinfer_cutlass_ms="
            f"{flashinfer_ms:.3f} min={flashinfer_min:.3f} max={flashinfer_max:.3f}"
        )
    if triton_ms is not None and flashinfer_ms is not None:
        print(f"speedup={triton_ms / flashinfer_ms:.3f}x")

    if args.check_output:
        if args.only != "both":
            raise ValueError("--check-output requires --only both")
        run_triton()
        run_flashinfer()
        torch.cuda.synchronize()
        diff = (triton_out.float() - flashinfer_out.float()).abs()
        denom = triton_out.float().abs().clamp_min(1e-3)
        print(f"max_abs_diff={diff.max().item():.6g}")
        print(f"max_rel_diff={(diff / denom).max().item():.6g}")


if __name__ == "__main__":
    main()
