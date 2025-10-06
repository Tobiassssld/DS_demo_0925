"""End-to-end pipeline for the user behavior demo."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .preprocess import PreprocessConfig, build_user_documents, preprocess
from .embeddings.word2vec_model import get_user_embeddings, train_word2vec
from .embeddings.doc2vec_model import build_tagged_documents, infer_user_vectors, train_doc2vec
from .embeddings.tfidf_model import tfidf_features
from .embeddings.autoencoder_model import AutoencoderConfig, train_autoencoder
from .clustering.kmeans_cluster import run_kmeans
from .analysis.business_metrics import category_preferences, compute_rfm_summary
from .analysis.insights_report import generate_html_report
from .analysis.model_comparison import compare_models, results_to_dataframe
from .visualization.tsne_plot import project_embeddings


def prepare_embeddings(
    weighted_transactions: pd.DataFrame,
    feature_table: pd.DataFrame,
    sequences: list[list[str]],
) -> Tuple[Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
    documents = build_user_documents(weighted_transactions)

    # Word2Vec embeddings
    w2v_model = train_word2vec(sequences, vector_size=16, epochs=100)
    word2vec_embeddings = np.array(get_user_embeddings(sequences, w2v_model))

    # Doc2Vec embeddings
    tagged_docs = build_tagged_documents(documents)
    d2v_model = train_doc2vec(tagged_docs, vector_size=16, epochs=80)
    doc2vec_df = infer_user_vectors(d2v_model, documents)

    # TF-IDF features
    tfidf_df, _ = tfidf_features(weighted_transactions, max_features=30)

    # Autoencoder on numeric features
    numeric = feature_table.drop(columns=["CustomerID"]).values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric)
    _, ae_embeddings = train_autoencoder(scaled, AutoencoderConfig(latent_dim=4, epochs=200))

    embeddings = {
        "word2vec": word2vec_embeddings,
        "doc2vec": doc2vec_df.drop(columns=["CustomerID"]).values,
        "tfidf": tfidf_df.drop(columns=["CustomerID"]).values,
        "autoencoder": ae_embeddings,
    }
    tables = {
        "doc2vec": doc2vec_df,
        "tfidf": tfidf_df,
    }
    return embeddings, tables


def align_customers(base_customers: list[str], table: pd.DataFrame) -> np.ndarray:
    mapping = table.set_index("CustomerID").reindex(base_customers)
    return mapping.values


def run_pipeline(
    data_dir: Path,
    reports_dir: Path,
    config: PreprocessConfig | None = None,
) -> Dict[str, Path]:
    transactions_path = data_dir / "OnlineRetail.csv"
    touchpoints_path = data_dir / "synthetic_touchpoints.csv"

    weighted, feature_table, sequences = preprocess(
        transactions_path=str(transactions_path),
        touchpoints_path=str(touchpoints_path),
        config=config,
    )

    embeddings, tables = prepare_embeddings(weighted, feature_table, sequences)

    customer_ids = feature_table["CustomerID"].tolist()
    embeddings["doc2vec"] = align_customers(customer_ids, tables["doc2vec"])
    embeddings["tfidf"] = align_customers(customer_ids, tables["tfidf"])

    def kmeans_clusterer(features: np.ndarray) -> np.ndarray:
        _, labels = run_kmeans(features, n_clusters=min(3, len(features)))
        return labels

    results = compare_models(
        embeddings=list(embeddings.values()),
        embedding_names=list(embeddings.keys()),
        clusterer=kmeans_clusterer,
    )
    metrics_df = results_to_dataframe(results)

    # Select first model for reporting
    selected_labels = results[0].labels
    cluster_mapping = pd.Series(selected_labels, index=customer_ids)
    cluster_feature_table = feature_table.copy()
    cluster_feature_table["cluster"] = selected_labels

    cluster_summary = compute_rfm_summary(cluster_feature_table)
    categories = category_preferences(weighted, cluster_mapping)

    projection = project_embeddings(embeddings["word2vec"], method="tsne", random_state=42)
    projection_df = pd.DataFrame(projection, columns=["x", "y"])
    projection_df["cluster"] = selected_labels

    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "summary_metrics.csv"
    cluster_profile_path = reports_dir / "cluster_profiles.csv"
    report_path = reports_dir / "insights_report.html"
    projection_path = reports_dir / "projection.csv"

    metrics_df.to_csv(summary_path, index=False)
    cluster_summary.to_csv(cluster_profile_path, index=False)
    projection_df.to_csv(projection_path, index=False)
    generate_html_report(metrics_df, cluster_summary, categories, report_path)

    return {
        "summary": summary_path,
        "clusters": cluster_profile_path,
        "report": report_path,
        "projection": projection_path,
    }


if __name__ == "__main__":  # pragma: no cover
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "data"
    reports_dir = base_dir / "reports"
    outputs = run_pipeline(data_dir=data_dir, reports_dir=reports_dir)
    print("Generated outputs:")
    for name, path in outputs.items():
        print(f"{name}: {path}")
