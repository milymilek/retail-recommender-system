from dataclasses import dataclass
from functools import cached_property
from typing import Any

import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset

from retail_recommender_system.layers.dotprod import DotProd
from retail_recommender_system.logging import init_logger
from retail_recommender_system.utils import approx_neg_sampl

logger = init_logger(__name__)


def collate_fn(batch):
    u_id = torch.cat([x["u_id"] for x in batch])
    i_id = torch.cat([x["i_id"] for x in batch])
    target = torch.cat([x["target"] for x in batch])
    return {"u_id": u_id, "i_id": i_id, "target": target}


def eval_collate_fn(batch):
    u_id = torch.cat([x["u_id"] for x in batch])
    return {"u_id": u_id}


class MFDataset(Dataset):
    def __init__(self, relations: pl.DataFrame, users: pl.DataFrame, items: pl.DataFrame, namings: dict[str, str], neg_sampl: int = 5):
        self._df = torch.from_numpy(relations.select(namings["user_id_map"], namings["item_id_map"]).to_numpy()).to(torch.int32)
        self._users = torch.from_numpy(users.get_column(namings["user_id_map"]).to_numpy()).to(torch.float32)
        self._items = torch.from_numpy(items.get_column(namings["item_id_map"]).to_numpy()).to(torch.float32)
        self._neg_sampl = neg_sampl

    @property
    def _n_users(self) -> int:
        return len(self._users)

    @property
    def _n_items(self) -> int:
        return len(self._items)

    @cached_property
    def users_set(self) -> torch.Tensor:
        return torch.arange(self._n_users, dtype=torch.int32)

    @cached_property
    def items_set(self) -> torch.Tensor:
        return torch.arange(self._n_items, dtype=torch.int32)

    @cached_property
    def ground_truth(self) -> torch.Tensor:
        return self._df.T

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._df[idx]
        user = row[0].unsqueeze(0)
        items = row[1].unsqueeze(0)

        u_id = user.repeat(self._neg_sampl + 1)
        i_id = torch.cat([items, approx_neg_sampl(self._n_items, self._neg_sampl)])
        target = torch.tensor([1.0] + [0.0] * self._neg_sampl, dtype=torch.float)

        return {"u_id": u_id, "i_id": i_id, "target": target}


class MFEvalDataset(IterableDataset):
    def __init__(self, base_dataset: MFDataset, user_batch_size: int):
        super().__init__()
        self._base_dataset = base_dataset
        self._user_batch_size = user_batch_size

    @property
    def users_set(self) -> torch.Tensor:
        return self._base_dataset.users_set

    @property
    def items_set(self) -> torch.Tensor:
        return self._base_dataset.items_set

    @property
    def ground_truth(self) -> torch.Tensor:
        return self._base_dataset.ground_truth

    def __len__(self):
        return len(self.users_set) // self._user_batch_size + 1

    def __iter__(self):
        for batch in self.users_set.split(self._user_batch_size):
            yield {"u_id": batch}


@dataclass
class MFModelConfig:
    n_users: int
    n_items: int
    emb_size: int
    dropout: float = 0.0


class MF(nn.Module):
    def __init__(self, config: MFModelConfig):
        super().__init__()
        self.user_factors = nn.Embedding(config.n_users, config.emb_size)
        self.item_factors = nn.Embedding(config.n_items, config.emb_size)

        self.dropout = nn.Dropout(config.dropout)

        self.dot = DotProd()

    def forward(self, x):
        user_factors = self.dropout(self.user_factors(x["u_id"]))
        item_factors = self.dropout(self.item_factors(x["i_id"]))
        return self.dot(user_factors, item_factors)

    @torch.no_grad()
    def recommend(self, x: dict[str, torch.Tensor]):
        user_emb = self.user_factors.weight[x["u_id"]]
        item_emb = self.item_factors.weight
        return torch.sigmoid(user_emb @ item_emb.T)
