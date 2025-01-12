from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from retail_recommender_system.data.datasets.base import BaseDataset
from retail_recommender_system.data.sparsifier import Sparsifier
from retail_recommender_system.utils import filter_set, split_by_time


class HMDataset(BaseDataset):
    @property
    def ds(self) -> str:
        return "hm"

    @property
    def n_users(self) -> int:
        return self.data["users"].get_column("customer_id_map").max() + 1  # type: ignore

    @property
    def n_items(self) -> int:
        return self.data["items"].get_column("article_id_map").max() + 1  # type: ignore

    @property
    def namings(self) -> dict[str, str]:
        return {
            "user_id": "customer_id",
            "item_id": "article_id",
            "user_id_map": "customer_id_map",
            "item_id_map": "article_id_map",
            "date": "t_dat",
        }

    def load_base(self) -> None:
        self.data = {
            "relations": pl.read_csv(self.base_path / "transactions_train.csv", try_parse_dates=True),
            "users": pl.read_csv(self.base_path / "customers.csv"),
            "items": pl.read_csv(self.base_path / "articles.csv"),
        }

    def process_base(self) -> None:
        # Filter by date
        relations = self.data["relations"].filter(pl.col("t_dat") > pl.lit(datetime(2020, 9, 1)))

        # # Filter out users with less than 5 transactions
        # filtered_users = self.data["relations"].group_by("customer_id").agg(pl.len()).filter(pl.col("len") >= 5).select("customer_id")
        # relations = self.data["relations"].join(filtered_users, on="customer_id", how="right")

        # Split data
        train_relations, valid_relations = split_by_time(relations, date_col=self.namings["date"], validation_ratio=0.3)
        valid_relations = filter_set(valid_relations, train_relations, self.namings["user_id"], self.namings["item_id"])

        # Filter users and items that do not appear in the train set
        uq_users = train_relations[self.namings["user_id"]].unique().to_numpy().ravel()
        users = self.data["users"].filter(pl.col("customer_id").is_in(uq_users))
        uq_items = train_relations[self.namings["item_id"]].unique().to_numpy().ravel()
        items = self.data["items"].filter(pl.col("article_id").is_in(uq_items))

        # Add path column to articles
        article_path_tuple_list = [(int(i.stem), str(i)) for i in (self.base_path / "images").rglob("*.jpg")]
        articles_path_map = pl.DataFrame(
            {self.namings["item_id"]: [i[0] for i in article_path_tuple_list], "path": [i[1] for i in article_path_tuple_list]}
        )
        items = items.join(articles_path_map, on=self.namings["item_id"], how="left")

        # Create a mapping from the original id to a new id
        users = users.with_columns(**{self.namings["user_id_map"]: pl.col(self.namings["user_id"]).cast(pl.Categorical).to_physical()})
        items = items.with_columns(
            **{self.namings["item_id_map"]: pl.col(self.namings["item_id"]).cast(pl.String).cast(pl.Categorical).to_physical()}
        )
        users_id_map = users.select(self.namings["user_id"], self.namings["user_id_map"]).unique()
        items_id_map = items.select(self.namings["item_id"], self.namings["item_id_map"]).unique()

        # Add mapping columns to relations
        train_relations = (
            train_relations.sort(self.namings["date"])
            .join(users_id_map, on=self.namings["user_id"], how="left")
            .join(items_id_map, on=self.namings["item_id"], how="left")
        )
        valid_relations = (
            valid_relations.sort(self.namings["date"])
            .join(users_id_map, on=self.namings["user_id"], how="left")
            .join(items_id_map, on=self.namings["item_id"], how="left")
        )

        self.data["users"] = users
        self.data["items"] = items
        self.data["relations"] = (train_relations, valid_relations)

    def save_intermediate(self, data=None, prefix=None) -> None:
        if data is None:
            data = self.data

        if prefix is not None:
            self._prefix = prefix

        self.intermediate.mkdir(parents=True, exist_ok=True)

        data["users"].write_parquet(self.intermediate / "customers.parquet")
        data["items"].write_parquet(self.intermediate / "articles.parquet")
        for i in range(len(data["relations"])):
            data["relations"][i].write_parquet(self.intermediate / f"transactions_{['train', 'valid'][i]}.parquet")

    def load(self) -> None:
        self.data = {
            "relations": (pl.read_parquet(self.intermediate / f"transactions_{i}.parquet") for i in ["train", "valid"]),
            "users": pl.read_parquet(self.intermediate / "customers.parquet"),
            "items": pl.read_parquet(self.intermediate / "articles.parquet"),
        }

    def build_sparsifier(self):
        return Sparsifier(self.data["relations"], self.data["users"], self.data["items"], namings=self.namings)
