from dataclasses import dataclass

import torch
import torch.nn as nn

from retail_recommender_system.logging import init_logger

logger = init_logger(__name__)


@dataclass
class DeepFMModelConfig:
    n_users: int
    n_items: int
    emb_size: int
    batch_norm: bool = False
    dropout: float = 0.0


class DeepFM(nn.Module):
    def __init__(self, config: DeepFMModelConfig):
        super().__init__()
        self.user_factors = nn.Embedding(config.n_users, config.emb_size)
        self.item_factors = nn.Embedding(config.n_items, config.emb_size)

        layers = []
        dims = [2 * config.emb_size] + [32, 16, 8]
        for in_dim, out_dim in zip(dims, dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=config.dropout))
        layers.append(nn.Linear(dims[-1], 1))

        self._layers = nn.Sequential(*layers)

    def forward(self, x):
        user_factors = self.user_factors(x["u_id"])
        item_factors = self.item_factors(x["i_id"])

        x = torch.cat([user_factors, item_factors], dim=1)
        return self._layers(x).squeeze()
