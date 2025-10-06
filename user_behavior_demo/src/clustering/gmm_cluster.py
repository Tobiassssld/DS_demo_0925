"""Gaussian Mixture clustering helpers."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.mixture import GaussianMixture


def run_gmm(features: np.ndarray, n_components: int = 3, random_state: int = 42) -> Tuple[GaussianMixture, np.ndarray]:
    model = GaussianMixture(n_components=n_components, covariance_type="full", random_state=random_state)
    labels = model.fit_predict(features)
    return model, labels


__all__ = ["run_gmm"]
