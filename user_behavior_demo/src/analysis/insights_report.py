"""Generate a lightweight HTML insight report from clustering outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

REPORT_TEMPLATE = """<html><head><title>Customer Insights</title></head><body>
<h1>Customer Segmentation Insights</h1>
<h2>Model Comparison</h2>
{model_table}
<h2>Cluster Summaries</h2>
{cluster_table}
<h2>Top Categories per Cluster</h2>
{category_table}
</body></html>"""


def generate_html_report(
    model_metrics: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    top_categories: pd.DataFrame,
    output_path: Path,
) -> Path:
    model_table = model_metrics.to_html(index=False, float_format="{:.3f}".format)
    cluster_table = cluster_summary.to_html(index=False, float_format="{:.2f}".format)
    category_table = top_categories.to_html(index=False)
    html = REPORT_TEMPLATE.format(
        model_table=model_table,
        cluster_table=cluster_table,
        category_table=category_table,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path


__all__ = ["generate_html_report"]
