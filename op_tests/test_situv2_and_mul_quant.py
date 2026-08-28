# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import dtypes
from aiter.ops.activation import situv2_and_mul_quant


def _situv2_ref(
    x: torch.Tensor,
    beta: float,
    linear_beta: float,
) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d].float()
    up = x[..., d:].float()
    return (
        beta
        * torch.tanh(gate / beta)
        * torch.sigmoid(gate)
        * (linear_beta * torch.tanh(up / linear_beta))
    )


# One case per host dispatch branch, plus a second token count on each of the
# two that Kimi-K3 actually runs.
@pytest.mark.parametrize(
    ("m", "d", "beta", "linear_beta"),
    [
        (8, 768, 4.0, 25.0),  # compile-time d, shared experts
        (129, 768, 2.5, 17.3),
        (5, 384, 4.0, 25.0),  # d <= 512, single pass in one wave
        (5, 1024, 4.0, 25.0),  # d <= 1024, VecSize 16 needs d % 16 == 0
        (5, 4224, 4.0, 25.0),  # runtime d, dense MLP
        (129, 4224, 6.0, 32.0),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm")
def test_situv2_and_mul_quant_ptpc(
    m: int,
    d: int,
    beta: float,
    linear_beta: float,
):
    torch.manual_seed(103 + d + m)
    x = torch.randn((m, 2 * d), device="cuda", dtype=torch.bfloat16)
    quantized = torch.empty((m, d), device="cuda", dtype=dtypes.fp8)
    scale = torch.empty((m, 1), device="cuda", dtype=torch.float32)

    situv2_and_mul_quant(
        quantized,
        x,
        scale,
        d,
        beta,
        linear_beta,
    )

    ref = _situv2_ref(x, beta, linear_beta)
    ref_scale = ref.abs().amax(dim=-1, keepdim=True) / torch.finfo(dtypes.fp8).max
    dequantized = quantized.float() * scale

    torch.testing.assert_close(scale, ref_scale, rtol=2e-5, atol=3e-8)
    torch.testing.assert_close(dequantized, ref, rtol=8e-2, atol=3e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm")
def test_situv2_and_mul_quant_zero_input():
    x = torch.zeros((4, 2 * 768), device="cuda", dtype=torch.bfloat16)
    quantized = torch.empty((4, 768), device="cuda", dtype=dtypes.fp8)
    scale = torch.empty((4, 1), device="cuda", dtype=torch.float32)

    situv2_and_mul_quant(quantized, x, scale, 768, 4.0, 25.0)

    assert torch.count_nonzero(quantized.float()) == 0
    assert torch.count_nonzero(scale) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm")
def test_situv2_and_mul_quant_empty_batch():
    x = torch.empty((0, 2 * 768), device="cuda", dtype=torch.bfloat16)
    quantized = torch.empty((0, 768), device="cuda", dtype=dtypes.fp8)
    scale = torch.empty((0, 1), device="cuda", dtype=torch.float32)

    situv2_and_mul_quant(quantized, x, scale, 768, 4.0, 25.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm")
def test_situv2_and_mul_quant_fullgraph():
    m, d = 4, 768
    x = torch.randn((m, 2 * d), device="cuda", dtype=torch.bfloat16)
    quantized = torch.empty((m, d), device="cuda", dtype=dtypes.fp8)
    scale = torch.empty((m, 1), device="cuda", dtype=torch.float32)

    @torch.compile(backend="eager", fullgraph=True)
    def compiled(out: torch.Tensor, inp: torch.Tensor, out_scale: torch.Tensor):
        situv2_and_mul_quant(out, inp, out_scale, d, 4.0, 25.0)
        return out, out_scale

    compiled(quantized, x, scale)
    ref = _situv2_ref(x, 4.0, 25.0)
    torch.testing.assert_close(quantized.float() * scale, ref, rtol=8e-2, atol=3e-1)
