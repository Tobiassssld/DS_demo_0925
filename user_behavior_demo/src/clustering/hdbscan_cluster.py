"""Wrapper around the HDBSCAN clustering algorithm."""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import hdbscan
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "hdbscan must be installed to use density based clustering"
    ) from exc


def run_hdbscan(
    features: np.ndarray,
    min_cluster_size: int = 3,
    metric: str = "euclidean",
    cluster_selection_epsilon: float = 0.0,
    prediction_data: bool = True,
    min_samples: Optional[int] = None,
) -> hdbscan.HDBSCAN:
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=metric,
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=prediction_data,
        min_samples=min_samples,
    )
    model.fit(features)
    return model


__all__ = ["run_hdbscan"]
