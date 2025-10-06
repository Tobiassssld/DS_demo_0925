"""Common clustering evaluation helpers."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import calinski_harabasz_score, silhouette_score


def evaluate_clustering(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    n_samples = len(labels)

    # Metrics such as silhouette and Calinski-Harabasz require at least two
    # clusters and strictly fewer clusters than samples.  Small demo datasets
    # occasionally violate these assumptions, so we guard against the runtime
    # errors by returning NaNs instead.
    if n_labels < 2 or n_labels >= n_samples:
        return {"silhouette": float("nan"), "calinski_harabasz": float("nan")}

    silhouette = silhouette_score(features, labels)
    calinski = calinski_harabasz_score(features, labels)
    return {"silhouette": float(silhouette), "calinski_harabasz": float(calinski)}


__all__ = ["evaluate_clustering"]
