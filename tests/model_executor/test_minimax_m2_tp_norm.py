# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.mamba import linear_attn
from vllm.model_executor.models import minimax_m2


class _DummyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise AssertionError("Test stub should not be executed.")


@pytest.fixture
def default_vllm_config():
    with set_current_vllm_config(VllmConfig()):
        yield


def test_minimax_text01_rmsnorm_tp_loader_handles_replicated_shards(
    default_vllm_config, monkeypatch
):
    monkeypatch.setattr(linear_attn, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(linear_attn, "get_tensor_model_parallel_rank", lambda: 3)

    norm = linear_attn.MiniMaxText01RMSNormTP(hidden_size=8, weight_shard_count=2)
    loaded_weight = torch.arange(8, dtype=norm.weight.dtype)

    norm.weight_loader(norm.weight, loaded_weight)
    assert torch.equal(norm.weight.detach(), loaded_weight[4:8])

    norm.weight.data.zero_()
    norm.weight_loader(norm.weight, loaded_weight[4:8].clone())
    assert torch.equal(norm.weight.detach(), loaded_weight[4:8])

    tp_local_concat = torch.cat(
        [
            loaded_weight[:4],
            loaded_weight[:4],
            loaded_weight[4:8],
            loaded_weight[4:8],
        ]
    )
    norm.weight.data.zero_()
    norm.weight_loader(norm.weight, tp_local_concat)
    assert torch.equal(norm.weight.detach(), loaded_weight[4:8])


def test_minimax_text01_rmsnorm_tp_loader_for_fully_replicated_kv(
    default_vllm_config,
    monkeypatch,
):
    monkeypatch.setattr(linear_attn, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(linear_attn, "get_tensor_model_parallel_rank", lambda: 2)

    norm = linear_attn.MiniMaxText01RMSNormTP(hidden_size=2, weight_shard_count=1)
    local_weight = torch.tensor([10.0, 11.0], dtype=norm.weight.dtype)
    tp_local_concat = local_weight.repeat(4)

    norm.weight_loader(norm.weight, tp_local_concat)
    assert torch.equal(norm.weight.detach(), local_weight)


def test_minimax_m2_attention_uses_local_kv_norm_width(
    default_vllm_config, monkeypatch
):
    monkeypatch.setattr(minimax_m2, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(linear_attn, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(linear_attn, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(minimax_m2, "QKVParallelLinear", _DummyModule)
    monkeypatch.setattr(minimax_m2, "RowParallelLinear", _DummyModule)
    monkeypatch.setattr(minimax_m2, "Attention", _DummyModule)
    monkeypatch.setattr(minimax_m2, "get_rope", lambda *args, **kwargs: _DummyModule())

    attn = minimax_m2.MiniMaxM2Attention(
        hidden_size=64,
        num_heads=8,
        num_kv_heads=1,
        rotary_dim=8,
        head_dim=8,
        max_position_embeddings=128,
    )

    assert attn.q_norm.weight.numel() == attn.q_size
    assert attn.k_norm.weight.numel() == attn.kv_size
