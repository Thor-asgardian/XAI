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

            probs_tensor: Tensor = out.squeeze().detach().cpu()
            probs_list = probs_tensor.tolist()

            if isinstance(probs_list, list):
                preds.extend(float(p) for p in probs_list)
            else:
                preds.append(float(probs_list))

                labels.extend(
                    int(v) for v in y.detach().cpu().int().tolist()
                )

                # ---- Compute metrics AFTER loop ----

                y_pred: List[int] = [1 if p > 0.5 else 0 for p in preds]

                report_raw = classification_report(labels, y_pred)
                report: str = report_raw if isinstance(report_raw, str) else str(report_raw)

                auc: float = float(roc_auc_score(labels, preds))

                print(report)
                print(f"ROC-AUC: {auc:.4f}")