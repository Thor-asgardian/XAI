from typing import Dict, Tuple
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from config import *
from models.model import MXMDF


BatchType = Tuple[Dict[str, Tensor], Tensor]


def train_model(
    train_data: Dataset[BatchType],
) -> nn.Module: # pyright: ignore[reportReturnType]

    model: nn.Module = MXMDF(INPUT_DIMS, LATENT_DIM)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    loader: DataLoader[BatchType] = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model.train()

    if EPOCHS <= 0:
        torch.save(model.state_dict(), "mxmdf_model.pth")
        return model

    for epoch in range(EPOCHS):
        total_loss: float = 0.0

        for inputs, labels in loader:
            preds, _ = model(inputs)

            loss: Tensor = criterion(
                preds.squeeze(),
                labels.float(),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

            torch.save(model.state_dict(), "mxmdf_model.pth")

            return model