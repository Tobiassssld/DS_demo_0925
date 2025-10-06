"""Utilities for cleaning user behavior data and constructing feature tables.

The preprocessing step loads the transactional dataset together with optional
marketing touch points, removes noise, and computes recency-aware interaction
weights that can be consumed by downstream embedding models.  The module keeps
functions small so that they can be reused by notebooks as well as the main
pipeline script.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PreprocessConfig:
    """Configuration options for the preprocessing stage."""

    recency_half_life: float = 30.0
    min_quantity: int = 1


def load_datasets(
    transactions_path: str,
    touchpoints_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load transactional and optional touchpoint datasets."""

    transactions = pd.read_csv(transactions_path, parse_dates=["InvoiceDate"])
    touchpoints = (
        pd.read_csv(touchpoints_path, parse_dates=["interaction_date"])
        if touchpoints_path
        else None
    )
    return transactions, touchpoints


def clean_transactions(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    """Clean raw transactions by removing returns and missing customers."""

    clean = df.copy()
    clean = clean[clean["Quantity"] >= config.min_quantity]
    clean = clean.dropna(subset=["CustomerID"])
    clean["CustomerID"] = clean["CustomerID"].astype(int).astype(str)
    clean["InvoiceDate"] = pd.to_datetime(clean["InvoiceDate"])
    clean["TotalPrice"] = clean["Quantity"] * clean["UnitPrice"]
    return clean


def compute_recency_weights(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    """Append a recency weight using an exponential decay."""

    max_date = df["InvoiceDate"].max()
    days_diff = (max_date - df["InvoiceDate"]).dt.days
    decay = np.exp(-np.log(2) * days_diff / config.recency_half_life)
    df = df.copy()
    df["RecencyWeight"] = decay
    df["WeightedTotal"] = df["TotalPrice"] * df["RecencyWeight"]
    return df


def build_user_sequences(df: pd.DataFrame) -> Tuple[List[List[str]], List[str]]:
    """Return product sequences ordered by time along with customer IDs."""

    df_sorted = df.sort_values("InvoiceDate")
    grouped = df_sorted.groupby(["CustomerID", "InvoiceNo"])
    sequences: List[List[str]] = []
    customers: List[str] = []
    for (customer_id, _), invoice in grouped:
        sequences.append(invoice["StockCode"].astype(str).tolist())
        customers.append(str(customer_id))
    return sequences, customers


def build_user_documents(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions per customer for document-level embeddings."""

    df_sorted = df.sort_values("InvoiceDate")
    agg = df_sorted.groupby("CustomerID").agg(
        {
            "Description": lambda x: " ".join(x.astype(str)),
            "WeightedTotal": "sum",
            "RecencyWeight": "mean",
        }
    )
    agg = agg.rename(columns={"Description": "doc"}).reset_index()
    return agg


def build_feature_table(
    df: pd.DataFrame,
    touchpoints: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Construct a RFM-like feature table for downstream clustering."""

    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        {
            "InvoiceDate": lambda x: (reference_date - x.max()).days,
            "InvoiceNo": "nunique",
            "WeightedTotal": "sum",
        }
    )
    rfm.columns = ["Recency", "Frequency", "Monetary"]

    if touchpoints is not None and not touchpoints.empty:
        touchpoints = touchpoints.copy()
        touchpoints["interaction_date"] = pd.to_datetime(
            touchpoints["interaction_date"]
        )
        channel_counts = (
            touchpoints.groupby(["CustomerID", "channel"])
            .size()
            .unstack(fill_value=0)
            .add_prefix("touch_")
        )
        rfm = rfm.join(channel_counts, how="left").fillna(0)
    return rfm.reset_index()


def preprocess(
    transactions_path: str,
    touchpoints_path: Optional[str] = None,
    config: Optional[PreprocessConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[List[str]], List[str]]:
    """Full preprocessing routine returning clean data and helper tables."""

    config = config or PreprocessConfig()
    transactions, touchpoints = load_datasets(transactions_path, touchpoints_path)
    clean = clean_transactions(transactions, config)
    weighted = compute_recency_weights(clean, config)
    sequences, sequence_customers = build_user_sequences(weighted)
    feature_table = build_feature_table(weighted, touchpoints)
    return weighted, feature_table, sequences, sequence_customers


__all__ = [
    "PreprocessConfig",
    "load_datasets",
    "clean_transactions",
    "compute_recency_weights",
    "build_user_sequences",
    "build_user_documents",
    "build_feature_table",
    "preprocess",
]
