"""Simple autoencoder implementation for dense feature learning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class AutoencoderConfig:
    latent_dim: int = 8
    hidden_dim: int = 32
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 1e-3
    device: str = "cpu"


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, config: AutoencoderConfig) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def train_autoencoder(features: np.ndarray, config: AutoencoderConfig) -> Tuple[Autoencoder, np.ndarray]:
    device = torch.device(config.device)
    model = Autoencoder(features.shape[1], config).to(device)
    dataset = TensorDataset(torch.from_numpy(features.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    model.train()
    for _ in range(config.epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        latent = model.encoder(torch.from_numpy(features.astype(np.float32)).to(device))
    return model, latent.cpu().numpy()


__all__ = ["Autoencoder", "AutoencoderConfig", "train_autoencoder"]
