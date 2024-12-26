from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from retail_recommender_system.evaluation.metrics import precision_k, recall_k
from retail_recommender_system.evaluation.prediction import recommend_k
from retail_recommender_system.logging import init_logger
from retail_recommender_system.training import History
from retail_recommender_system.utils import batch_dict_to_device, save_model

logger = init_logger(__name__)


def collate_fn(batch):
    u_id = torch.cat([x["u_id"] for x in batch])
    i_id = torch.cat([x["i_id"] for x in batch])
    target = torch.cat([x["target"] for x in batch])
    return {"u_id": u_id, "i_id": i_id, "target": target}


def eval_collate_fn(batch):
    batch = torch.cat(batch)
    return {"u_id": batch[:, 0], "i_id": batch[:, 1]}


class MFDataset(Dataset):
    def __init__(self, df: pl.DataFrame, n_items: int, neg_sampl: int = 5):
        self._df = df
        self._n_items = n_items
        self._neg_sampl = neg_sampl

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._df[idx]
        user = torch.tensor(row.get_column("customer_id_map").to_numpy(), dtype=torch.int32)
        items = torch.tensor(row.get_column("article_id_map").to_numpy(), dtype=torch.int32)

        u_id = user.repeat(self._neg_sampl + 1)
        i_id = torch.cat([items, self._approx_neg_sampl()])
        target = torch.tensor([1.0] + [0.0] * self._neg_sampl, dtype=torch.float)

        return {"u_id": u_id, "i_id": i_id, "target": target}

    def _approx_neg_sampl(self):
        neg_i_id = torch.randint(low=0, high=self._n_items, size=(self._neg_sampl,), dtype=torch.int32)
        return neg_i_id

    def users_set(self):
        return torch.from_numpy(self._df.select("customer_id_map").unique().sort(by="customer_id_map").to_numpy().flatten()).to(torch.int32)

    def items_set(self):
        return torch.from_numpy(self._df.select("article_id_map").unique().sort(by="article_id_map").to_numpy().flatten()).to(torch.int32)

    def ground_truth(self):
        return torch.from_numpy(self._df.select("customer_id_map", "article_id_map").to_numpy()).to(torch.int32).T


class MFEvalDataset(IterableDataset):
    def __init__(self, users_set: torch.Tensor, items_set: torch.Tensor, user_batch_size: int):
        super().__init__()
        self.users_set = users_set
        self.items_set = items_set

        self._user_batch_size = user_batch_size

    @property
    def _n_items(self):
        return self.items_set.shape[0]

    def get_batch_data(self, batch):
        u_id = torch.repeat_interleave(batch, self._n_items)
        i_id = self.items_set.repeat(len(batch))

        return torch.column_stack((u_id, i_id))

    def __len__(self):
        return len(self.users_set) // self._user_batch_size + 1

    def __iter__(self):
        for batch in self.users_set.split(self._user_batch_size):
            yield self.get_batch_data(batch)


@dataclass
class MFModelConfig:
    n_users: int
    n_items: int
    emb_size: int


class MF(nn.Module):
    def __init__(self, config: MFModelConfig):
        super().__init__()
        self.user_factors = nn.Embedding(config.n_users, config.emb_size)
        self.item_factors = nn.Embedding(config.n_items, config.emb_size)

    def forward(self, x):
        user_factors = self.user_factors(x["u_id"])
        item_factors = self.item_factors(x["i_id"])
        return (user_factors * item_factors).sum(1)
