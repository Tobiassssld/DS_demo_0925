"""Baseline TF-IDF vectorization of customer product descriptions."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def tfidf_features(df: pd.DataFrame, max_features: int = 200) -> Tuple[pd.DataFrame, TfidfVectorizer]:
    texts = df.groupby("CustomerID")["Description"].apply(lambda x: " ".join(x))
    vectorizer = TfidfVectorizer(max_features=max_features)
    matrix = vectorizer.fit_transform(texts)
    feature_df = pd.DataFrame(
        matrix.toarray(),
        index=texts.index,
        columns=[f"tfidf_{f}" for f in vectorizer.get_feature_names_out()],
    )
    feature_df = feature_df.reset_index().rename(columns={"CustomerID": "CustomerID"})
    return feature_df, vectorizer


__all__ = ["tfidf_features"]
