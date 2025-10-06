"""Doc2Vec representation of aggregated user documents."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd


def build_tagged_documents(df: pd.DataFrame) -> List[TaggedDocument]:
    documents: List[TaggedDocument] = []
    for _, row in df.iterrows():
        tokens = row["doc"].split()
        documents.append(TaggedDocument(words=tokens, tags=[row["CustomerID"]]))
    return documents


def train_doc2vec(
    documents: Iterable[TaggedDocument],
    vector_size: int = 64,
    epochs: int = 40,
    save_path: Path | None = None,
) -> Doc2Vec:
    model = Doc2Vec(vector_size=vector_size, min_count=1, workers=1)
    model.build_vocab(documents)
    model.train(documents, total_examples=len(documents), epochs=epochs)
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
    return model


def infer_user_vectors(model: Doc2Vec, df: pd.DataFrame) -> pd.DataFrame:
    vectors = []
    for _, row in df.iterrows():
        vector = model.infer_vector(row["doc"].split())
        vectors.append(vector)
    vector_df = pd.DataFrame(vectors, columns=[f"doc2vec_{i}" for i in range(model.vector_size)])
    vector_df.insert(0, "CustomerID", df["CustomerID"].values)
    return vector_df


__all__ = ["build_tagged_documents", "train_doc2vec", "infer_user_vectors"]
