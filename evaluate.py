from typing import Dict, Tuple, List
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score


BatchType = Tuple[Dict[str, Tensor], Tensor]


def evaluate(
    model: nn.Module,
    dataloader: DataLoader[BatchType],
) -> None:

    model.eval()

    preds: List[float] = []
    labels: List[int] = []

    with torch.no_grad():
        for inputs, y in dataloader:
            out, _ = model(inputs)
            probs = out.squeeze().detach().cpu().tolist()

            preds.extend(probs if isinstance(probs, list) else [probs])
            labels.extend(y.detach().cpu().int().tolist())

            y_pred: List[int] = [1 if p > 0.5 else 0 for p in preds]

            report: str = classification_report(labels, y_pred)
            auc: float = roc_auc_score(labels, preds)

            print(report)
            print(f"ROC-AUC: {auc:.4f}")