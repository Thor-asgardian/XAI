from typing import Any
import shap
import numpy as np


def shap_explain(model: Any, fused_features: np.ndarray) -> None:
    explainer = shap.Explainer(model)
    shap_values = explainer(fused_features)
    shap.plots.bar(shap_values)