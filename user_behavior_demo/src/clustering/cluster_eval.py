"""Common clustering evaluation helpers."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import calinski_harabasz_score, silhouette_score


def evaluate_clustering(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    if len(set(labels)) <= 1:
        return {"silhouette": float("nan"), "calinski_harabasz": float("nan")}
    silhouette = silhouette_score(features, labels)
    calinski = calinski_harabasz_score(features, labels)
    return {"silhouette": float(silhouette), "calinski_harabasz": float(calinski)}


__all__ = ["evaluate_clustering"]
