from typing import Dict
import torch
from torch import nn, Tensor


def generate_counterfactual(
    model: nn.Module,
    sample: Dict[str, Tensor],
    epsilon: float = 0.05,
) -> Dict[str, Tensor]: # type: ignore

    model.eval()

    sample_cf: Dict[str, Tensor] = {
        k: v.clone().detach().requires_grad_(True)
        for k, v in sample.items()
    }

    output, _ = model(sample_cf)
    loss: Tensor = -torch.log(output)

    loss.backward()

    cf: Dict[str, Tensor] = {}

    for k, v in sample_cf.items():
        if v.grad is None:
            raise RuntimeError("Gradient not computed")

        cf[k] = v - epsilon * v.grad.sign()

        return cf