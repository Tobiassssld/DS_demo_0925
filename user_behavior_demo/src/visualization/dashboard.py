"""Streamlit dashboard to explore segmentation results."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .cluster_radar import build_radar_chart
from .tsne_plot import plot_projection


def run_dashboard(
    model_metrics: pd.DataFrame,
    projection: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    top_categories: pd.DataFrame,
) -> None:
    st.title("Customer Segmentation Demo")
    st.subheader("Model Comparison")
    st.dataframe(model_metrics)

    st.subheader("Embedding Projection")
    st.plotly_chart(plot_projection(projection[["x", "y"]].values, projection["cluster"].values))

    st.subheader("Cluster Summary")
    st.plotly_chart(build_radar_chart(cluster_summary))

    st.subheader("Top Categories")
    st.dataframe(top_categories)


__all__ = ["run_dashboard"]
