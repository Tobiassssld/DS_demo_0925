"""Train a Word2Vec model on purchase sequences."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from gensim.models import Word2Vec


def train_word2vec(
    sequences: Sequence[Sequence[str]],
    vector_size: int = 32,
    window: int = 5,
    min_count: int = 1,
    sg: int = 1,
    epochs: int = 50,
    save_path: Path | None = None,
) -> Word2Vec:
    """Train a Word2Vec model on the provided customer purchase sequences."""

    model = Word2Vec(
        sentences=list(sequences),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        epochs=epochs,
        workers=1,
    )
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
    return model


def get_user_embeddings(
    sequences: Sequence[Sequence[str]], model: Word2Vec
) -> List[List[float]]:
    """Average product vectors for each sequence to form user embeddings."""

    embeddings: List[List[float]] = []
    for sequence in sequences:
        vectors = [model.wv[token] for token in sequence if token in model.wv]
        if vectors:
            embeddings.append(list(sum(vectors) / len(vectors)))
        else:
            embeddings.append([0.0] * model.vector_size)
    return embeddings


__all__ = ["train_word2vec", "get_user_embeddings"]
