# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm.lora.ops.triton_ops import lora_shrink_op


class _FakeKernel:
    def __init__(self) -> None:
        self.inputs_dtype: torch.dtype | None = None

    def __getitem__(self, grid):
        def launch(inputs, *args, **kwargs):
            self.inputs_dtype = inputs.dtype

        return launch


@pytest.mark.skip_global_cleanup
def test_lora_shrink_casts_inputs_to_lora_weight_dtype(monkeypatch):
    fake_kernel = _FakeKernel()
    monkeypatch.setattr(lora_shrink_op, "_lora_shrink_kernel", fake_kernel)
    monkeypatch.setattr(
        lora_shrink_op,
        "_get_lora_a_ptr",
        lambda weights, device: (torch.tensor([0], device=device), 0, 0, 0),
    )
    monkeypatch.setattr(
        lora_shrink_op,
        "get_lora_op_configs",
        lambda *args, **kwargs: {
            "block_m": 1,
            "block_n": 1,
            "block_k": 1,
            "split_k": 1,
            "num_warps": 1,
            "num_stages": 1,
            "num_ctas": 1,
        },
    )
    monkeypatch.setattr(lora_shrink_op, "supports_pdl", lambda device: False)
    monkeypatch.setattr(
        lora_shrink_op.triton, "cdiv", lambda a, b: math.ceil(a / b), raising=False
    )

    inputs = torch.randn(2, 4, dtype=torch.bfloat16)
    lora_a = [torch.randn(1, 3, 4, dtype=torch.float16)]
    output = torch.empty((1, 2, 3), dtype=torch.float32)
    token_mapping = torch.tensor([0, 0], dtype=torch.long)
    token_indices = torch.tensor([0, 1], dtype=torch.long)
    num_tokens_per_lora = torch.tensor([2], dtype=torch.long)
    lora_token_start_loc = torch.tensor([0, 2], dtype=torch.long)
    lora_ids = torch.tensor([0], dtype=torch.long)
    no_lora_flag_cpu = torch.tensor([False])
    num_active_loras = torch.tensor([1], dtype=torch.long)

    lora_shrink_op._lora_shrink(
        inputs,
        lora_a,
        output,
        token_mapping,
        token_indices,
        num_tokens_per_lora,
        lora_token_start_loc,
        lora_ids,
        no_lora_flag_cpu,
        num_active_loras,
        1.0,
    )

    assert fake_kernel.inputs_dtype == torch.float16
