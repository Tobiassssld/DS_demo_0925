"""Utility functions for running KMeans clustering."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def run_kmeans(features: np.ndarray, n_clusters: int = 3, random_state: int = 42) -> Tuple[KMeans, np.ndarray]:
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(features)
    return model, labels


def attach_cluster_labels(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    output = df.copy()
    output["cluster"] = labels
    return output


__all__ = ["run_kmeans", "attach_cluster_labels"]
