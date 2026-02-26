import torch
from models.model import MXMDF
from config import INPUT_DIMS, LATENT_DIM


def test_forward_pass() -> None:
    model = MXMDF(INPUT_DIMS, LATENT_DIM)
    model.eval()

    batch_size = 2

    sample = {
        name: torch.randn(batch_size, dim)
        for name, dim in INPUT_DIMS.items()
    }

    output, attn = model(sample)

    assert output.shape[0] == batch_size
    assert attn.shape[0] == batch_size