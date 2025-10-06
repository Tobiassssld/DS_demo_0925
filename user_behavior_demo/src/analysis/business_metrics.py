"""Business metric utilities for customer clustering outputs."""
from __future__ import annotations

from typing import Dict

import pandas as pd


def compute_rfm_summary(feature_table: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    metrics = feature_table.groupby(cluster_col)[["Recency", "Frequency", "Monetary"]].agg(["mean", "median"])
    metrics.columns = ["_".join(col).strip() for col in metrics.columns.values]
    return metrics.reset_index()


def retention_by_cluster(transactions: pd.DataFrame, cluster_mapping: pd.Series) -> pd.DataFrame:
    df = transactions.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    cluster_df = df.merge(
        cluster_mapping.rename("cluster"),
        left_on="CustomerID",
        right_index=True,
        how="left",
    )
    retention = (
        cluster_df.groupby(["cluster", "YearMonth"])["CustomerID"]
        .nunique()
        .groupby(level=0)
        .apply(lambda x: x / x.max())
        .reset_index(name="retention_rate")
    )
    return retention


def category_preferences(transactions: pd.DataFrame, cluster_mapping: pd.Series) -> pd.DataFrame:
    df = transactions.copy()
    cluster_df = df.merge(
        cluster_mapping.rename("cluster"),
        left_on="CustomerID",
        right_index=True,
        how="left",
    )
    category_pref = (
        cluster_df.groupby(["cluster", "Description"])["Quantity"].sum().reset_index()
    )
    top_categories = (
        category_pref.sort_values(["cluster", "Quantity"], ascending=[True, False])
        .groupby("cluster")
        .head(3)
        .reset_index(drop=True)
    )
    return top_categories


__all__ = [
    "compute_rfm_summary",
    "retention_by_cluster",
    "category_preferences",
]
