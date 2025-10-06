"""t-SNE/UMAP projection utilities."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

try:
    import umap
except ImportError:  # pragma: no cover - optional dependency
    umap = None


def project_embeddings(
    features: np.ndarray,
    method: str = "tsne",
    random_state: int = 42,
    n_components: int = 2,
    **kwargs,
) -> np.ndarray:
    if method == "umap" and umap is not None:
        reducer = umap.UMAP(n_components=n_components, random_state=random_state, **kwargs)
        return reducer.fit_transform(features)
    n_samples = features.shape[0]
    if n_samples <= 1:
        raise ValueError("At least two samples are required for projection.")

    perplexity = kwargs.pop("perplexity", 30)
    max_valid_perplexity = max(1, n_samples - 1)
    if perplexity >= n_samples:
        perplexity = min(perplexity, max_valid_perplexity)

    tsne = TSNE(
        n_components=n_components,
        random_state=random_state,
        perplexity=perplexity,
        **kwargs,
    )
    return tsne.fit_transform(features)


def plot_projection(
    embeddings: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Embedding Projection",
) -> px.scatter:
    df = pd.DataFrame(embeddings, columns=["x", "y"])
    if labels is not None:
        df["cluster"] = labels
    fig = px.scatter(df, x="x", y="y", color="cluster" if labels is not None else None, title=title)
    return fig


__all__ = ["project_embeddings", "plot_projection"]
