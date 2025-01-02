from dataclasses import dataclass

import torch
import torch.nn as nn

from retail_recommender_system.layers.ff import FeedForward
from retail_recommender_system.logging import init_logger

logger = init_logger(__name__)


@dataclass
class NCFModelConfig:
    n_users: int
    n_items: int
    emb_size: int
    ff_out: int = 16
    dropout: float = 0.0


class NCF(nn.Module):
    def __init__(self, config: NCFModelConfig):
        super().__init__()
        self.user_factors = nn.Embedding(config.n_users, config.emb_size)
        self.item_factors = nn.Embedding(config.n_items, config.emb_size)

        self.dropout = nn.Dropout(config.dropout)

        self.ff = FeedForward(sizes=[config.emb_size * 2, 64, 32, config.ff_out], batch_norm=True, dropout=config.dropout)
        self.lin = nn.Linear(config.ff_out + config.emb_size, 1)

    def forward(self, x):
        user_factors = self.dropout(self.user_factors(x["u_id"]))
        item_factors = self.dropout(self.item_factors(x["i_id"]))
        mlp_signal = self.ff(torch.cat([user_factors, item_factors], dim=1))
        gmf_signal = user_factors * item_factors
        return self.lin(torch.cat([mlp_signal, gmf_signal], dim=1)).squeeze(1)
