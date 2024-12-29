from dataclasses import dataclass
from functools import cached_property
from typing import Any

import polars as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, IterableDataset

from retail_recommender_system.logging import init_logger

logger = init_logger(__name__)


def collate_fn(batch):
    u_id = torch.cat([x["u_id"] for x in batch])
    i_id = torch.cat([x["i_id"] for x in batch])
    u_attr = torch.cat([x["u_attr"] for x in batch])
    i_attr = torch.cat([x["i_attr"] for x in batch])
    target = torch.cat([x["target"] for x in batch])
    return {"u_id": u_id, "i_id": i_id, "u_attr": u_attr, "i_attr": i_attr, "target": target}


class DeepFMDataset(Dataset):
    def __init__(self, relations: pl.DataFrame, users: pl.DataFrame, items: pl.DataFrame, neg_sampl: int = 5):
        self._df = relations
        self._users = self._process_users(users)
        self._items = self._process_items(items)
        self._neg_sampl = neg_sampl

    @property
    def _n_users(self) -> int:
        return len(self._users)

    @property
    def _n_items(self) -> int:
        return len(self._items)

    @property
    def user_attr_size(self):
        return self._users.shape[1]

    @property
    def item_attr_size(self):
        return self._items.shape[1]

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._df[idx]
        user = torch.tensor(row.get_column("customer_id_map").to_numpy(), dtype=torch.int32)
        items = torch.tensor(row.get_column("article_id_map").to_numpy(), dtype=torch.int32)

        user_attr = self.get_users_attr(user.repeat(self._neg_sampl + 1, 1))
        item_attr = self.get_items_attr(items.repeat(self._neg_sampl + 1, 1))

        u_id = user.repeat(self._neg_sampl + 1)
        i_id = torch.cat([items, self._approx_neg_sampl()])
        target = torch.tensor([1.0] + [0.0] * self._neg_sampl, dtype=torch.float)

        return {"u_id": u_id, "i_id": i_id, "u_attr": user_attr, "i_attr": item_attr, "target": target}

    def get_users_attr(self, idx: int | torch.Tensor) -> torch.Tensor:
        if isinstance(idx, int):
            idx = torch.tensor([idx], dtype=torch.int32)

        return self._users[idx]

    def get_items_attr(self, idx: int | torch.Tensor) -> torch.Tensor:
        if isinstance(idx, int):
            idx = torch.tensor([idx], dtype=torch.int32)

        return self._items[idx]

    def _approx_neg_sampl(self):
        neg_i_id = torch.randint(low=0, high=self._n_items, size=(self._neg_sampl,), dtype=torch.int32)
        return neg_i_id

    def _process_users(self, users: pl.DataFrame) -> torch.Tensor:
        # sort by customer_id_map
        users = users.sort("customer_id_map")

        # imput values
        users_imputed = users.with_columns(
            pl.col("FN").fill_null(0.0),
            pl.col("Active").fill_null(0.0),
            pl.col("age").fill_null(strategy="mean"),
            pl.col("club_member_status").fill_null("ACTIVE"),
            pl.col("fashion_news_frequency").fill_null("NONE"),
        )

        # one-hot encode categorical columns
        categorical_columns = ["club_member_status", "fashion_news_frequency"]

        encoder = OneHotEncoder()
        encoded_data = encoder.fit_transform(users_imputed.select(categorical_columns).to_pandas())
        schema_cols = [f"{col}_{cat}" for col, cats in zip(categorical_columns, encoder.categories_) for cat in cats]
        encoded_df = pl.DataFrame(encoded_data.toarray(), schema=schema_cols)
        users_imputed_merged = pl.concat([users_imputed, encoded_df], how="horizontal")

        return torch.tensor(users_imputed_merged.select("FN", "Active", "age", *schema_cols).to_numpy(), dtype=torch.float32)

    def _process_items(self, items: pl.DataFrame) -> torch.Tensor:
        # sort by article_id_map
        items = items.sort("article_id_map")

        # one-hot encode categorical columns
        categorical_columns = [
            "product_type_name",
            "product_group_name",
            "graphical_appearance_name",
            "colour_group_name",
            "perceived_colour_value_name",
            "perceived_colour_master_name",
            "department_name",
            "index_name",
            "index_group_name",
            "section_name",
            "garment_group_name",
        ]

        encoder = OneHotEncoder()
        encoded_data = encoder.fit_transform(items.select(categorical_columns).to_pandas())
        schema_cols = [f"{col}_{cat}" for col, cats in zip(categorical_columns, encoder.categories_) for cat in cats]
        encoded_df = pl.DataFrame(encoded_data.toarray(), schema=schema_cols)
        items_merged = pl.concat([items, encoded_df], how="horizontal")

        return torch.tensor(items_merged.select(*schema_cols).to_numpy(), dtype=torch.float32)

    def users_set(self) -> torch.Tensor:
        return torch.arange(self._n_users, dtype=torch.int32)

    def items_set(self) -> torch.Tensor:
        return torch.arange(self._n_items, dtype=torch.int32)

    def ground_truth(self) -> torch.Tensor:
        return torch.from_numpy(self._df.select("customer_id_map", "article_id_map").to_numpy()).to(torch.int32).T


class DeepFMEvalDataset(IterableDataset):
    def __init__(self, base_dataset: DeepFMDataset, user_batch_size: int = 1024):
        super().__init__()

        self._base_dataset = base_dataset
        self._user_batch_size = user_batch_size

    @cached_property
    def users_set(self) -> torch.Tensor:
        return self._base_dataset.users_set()

    @cached_property
    def items_set(self) -> torch.Tensor:
        return self._base_dataset.items_set()

    @cached_property
    def ground_truth(self) -> torch.Tensor:
        return self._base_dataset.ground_truth()

    def get_batch_data(self, batch) -> dict[str, torch.Tensor]:
        u_id = torch.repeat_interleave(batch, self._base_dataset._n_items)
        i_id = self.items_set.repeat(len(batch))

        user_attr = self._base_dataset.get_users_attr(u_id)
        item_attr = self._base_dataset.get_items_attr(i_id)

        return {
            "u_id": u_id,
            "i_id": i_id,
            "u_attr": user_attr,
            "i_attr": item_attr,
            "target": torch.zeros(len(u_id), dtype=torch.float),
        }

    def __len__(self):
        return self._base_dataset._n_users // self._user_batch_size + 1

    def __iter__(self):
        for batch in self.users_set.split(self._user_batch_size):
            yield self.get_batch_data(batch)


@dataclass
class DeepFMModelConfig:
    n_users: int
    n_items: int
    user_attr_size: int
    item_attr_size: int
    emb_size: int
    layer_sizes: list[int]
    batch_norm: bool = False
    dropout: float = 0.0


class DeepFM(nn.Module):
    def __init__(self, config: DeepFMModelConfig):
        super().__init__()
        self.user_factors = nn.Embedding(config.n_users, config.emb_size)
        self.item_factors = nn.Embedding(config.n_items, config.emb_size)

        layers = []
        dims = [2 * config.emb_size + config.user_attr_size + config.item_attr_size] + [32]
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

        user_attr = x["u_attr"].squeeze(1)
        item_attr = x["i_attr"].squeeze(1)

        x = torch.cat([user_factors, item_factors, user_attr, item_attr], dim=1)
        return self._layers(x).squeeze()
