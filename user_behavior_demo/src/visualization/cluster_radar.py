"""Radar chart visualization for cluster profiles."""
from __future__ import annotations

import pandas as pd
import plotly.express as px


def build_radar_chart(cluster_summary: pd.DataFrame) -> px.line_polar:
    melted = cluster_summary.melt(id_vars="cluster", var_name="metric", value_name="value")
    fig = px.line_polar(melted, r="value", theta="metric", color="cluster", line_close=True)
    fig.update_traces(fill="toself")
    return fig


__all__ = ["build_radar_chart"]
