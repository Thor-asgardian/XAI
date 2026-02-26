from typing import Dict, Tuple
import torch
from torch import nn, Tensor
from captum.attr import IntegratedGradients


class IGWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

def forward(
    self,
    static: Tensor,
    dynamic: Tensor,
    network: Tensor,
    memory: Tensor,
    entropy: Tensor,
) -> Tensor:

    inputs: Dict[str, Tensor] = {
        "static": static,
        "dynamic": dynamic,
        "network": network,
        "memory": memory,
        "entropy": entropy,
    }

    out, _ = self.model(inputs)
    return out


def explain_with_ig(
    model: nn.Module,
    sample: Dict[str, Tensor],
) -> Tuple[Tensor, ...]:

    wrapper = IGWrapper(model)
    ig = IntegratedGradients(wrapper)

    attributions = ig.attribute(
        (
            sample["static"],
            sample["dynamic"],
            sample["network"],
            sample["memory"],
            sample["entropy"],
        ),
        target=0,
    )

    return attributions