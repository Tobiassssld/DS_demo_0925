"""Compare embedding and clustering combinations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd

from ..clustering.cluster_eval import evaluate_clustering


@dataclass
class ModelResult:
    name: str
    labels: np.ndarray
    metrics: Dict[str, float]


def compare_models(
    embeddings: Sequence[np.ndarray],
    embedding_names: Sequence[str],
    clusterer: Callable[[np.ndarray], np.ndarray],
) -> List[ModelResult]:
    results: List[ModelResult] = []
    for emb, name in zip(embeddings, embedding_names):
        labels = clusterer(emb)
        metrics = evaluate_clustering(emb, labels)
        results.append(ModelResult(name=name, labels=labels, metrics=metrics))
    return results


def results_to_dataframe(results: Sequence[ModelResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        row = {"model": result.name}
        row.update(result.metrics)
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["ModelResult", "compare_models", "results_to_dataframe"]
