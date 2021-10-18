# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import logging

import pytest
import torch
from torch.cuda.amp.autocast_mode import autocast

from xformers.components.residual import LayerNormStyle, PostNorm, PreNorm

try:
    from xformers.triton import ResidualLayerNorm
    from xformers.triton.utils import gpu_capabilities_older_than_70

    _triton_available = True
except ImportError:
    logging.warning("Triton is not available, some optimizations will not be tested.")
    _triton_available = False

# Testing odd shapes on purpose
SHAPES = [
    (384, 128),
    (8, 384, 128),
    (8, 784, 512),
    (4, 2048, 384),
    (4, 3136, 1024),
    (2, 1024, 2048),
    (2, 2048, 4096),
    (2, 4096, 4096),
    (1, 2048, 12288),
]


class Plus(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


@pytest.mark.skipif(not _triton_available, reason="Triton is not available")
@pytest.mark.skipif(
    not _triton_available or gpu_capabilities_older_than_70(),
    reason="Triton requires a SM70+ GPU",
)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("style", [LayerNormStyle.Pre, LayerNormStyle.Post])
def test_residual_layernorm_parity(shape, amp, style):
    """Check that PyTorch and Triton softmax give the same result"""

    # Get the same inputs
    torch.random.manual_seed(0)
    X = torch.normal(0, 1, size=shape, device="cuda", requires_grad=True)

    torch.random.manual_seed(0)
    X_ = torch.normal(0, 1, size=shape, device="cuda", requires_grad=True)

    # Initialize the two layers, weights are 1 and 0 by default, no randomness
    torch_res_layernorm: torch.nn.Module = (
        PreNorm(d_model=X.shape[-1], sublayer=Plus(), use_triton=False)
        if style == LayerNormStyle.Pre
        else PostNorm(d_model=X.shape[-1], sublayer=Plus(), use_triton=False)
    )
    torch_res_layernorm = torch_res_layernorm.cuda()

    # pyre-ignore[16]: TODO(T101400990): Pyre did not recognize the
    # `FusedLinearNorm` import.
    # Fused layernorm equivalent
    triton_res_layernorm = ResidualLayerNorm(
        layer_norm_style=style, normalized_shape=X.shape[-1]
    )
    triton_res_layernorm = triton_res_layernorm.cuda()

    with autocast(enabled=amp):
        assert torch.allclose(X, X_)  # sanity checking, else all hell breaks loose

        # Check the forward pass
        y_torch = torch_res_layernorm(X, X)
        y_triton = triton_res_layernorm(X_, X_)
        assert torch.allclose(
            y_torch.norm(), y_triton.norm(), atol=1e-3
        ), f"{torch.norm(y_torch)} vs. {torch.norm(y_triton)}"

        # Check that BW also gives the same result
        loss_torch = torch.norm(y_torch)
        loss_torch.backward()

        loss_triton = torch.norm(y_triton)
        loss_triton.backward()

        print(torch.norm(y_torch), torch.norm(y_triton))

        print(y_torch[0, :])
        print(y_triton[0, :])

        # There are 3 items to check:
        # - gradient on the inputs
        assert torch.allclose(
            X.grad, X_.grad
        ), f"Inputs grad mismatch: {torch.norm(X.grad)} vs. {torch.norm(X_.grad)}"

        # - gradient on the layernorm weight
        assert torch.allclose(
            torch_res_layernorm.weight.grad, triton_res_layernorm.weight.grad, atol=1e-3
        ), (
            f"Weight grad mismatch: {torch.norm(torch_res_layernorm.weight.grad)} vs."
            + f" {torch.norm(triton_res_layernorm.weight.grad)}"
        )

        # - gradient on the layernorm bias
        assert torch.allclose(
            torch_res_layernorm.bias.grad, triton_res_layernorm.bias.grad, atol=1e-3
        ), (
            f"Bias grad mismatch: {torch.norm(torch_res_layernorm.bias.grad)} vs."
            + f" {torch.norm(triton_res_layernorm.bias.grad)}"
        )
