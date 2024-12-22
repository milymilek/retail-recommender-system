from typing import Any

import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset


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


def collate_fn(batch):
    u_id = torch.cat([x["u_id"] for x in batch])
    i_id = torch.cat([x["i_id"] for x in batch])
    target = torch.cat([x["target"] for x in batch])
    return {"u_id": u_id, "i_id": i_id, "target": target}


class MF(nn.Module):
    def __init__(self, n_users, n_items, emb_size):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, emb_size)
        self.item_factors = nn.Embedding(n_items, emb_size)

    def forward(self, x):
        user_factors = self.user_factors(x["u_id"])
        item_factors = self.item_factors(x["i_id"])
        return (user_factors * item_factors).sum(1)
