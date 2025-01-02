from pathlib import Path

import polars as pl

from retail_recommender_system.data.datasets.base import BaseDataset
from retail_recommender_system.data.sparsifier import Sparsifier


class HMDataset(BaseDataset):
    @property
    def ds(self) -> str:
        return "hm"

    def load_base(self):
        self.data = {
            "relations": pl.read_csv(".data/hm/base/transactions_train.csv", try_parse_dates=True),
            "users": pl.read_csv(".data/hm/base/customers.csv"),
            "items": pl.read_csv(".data/hm/base/articles.csv"),
        }

    def process_base(self):
        # Filter out users with less than 5 transactions
        filtered_users = self.data["relations"].group_by("customer_id").agg(pl.len()).filter(pl.col("len") >= 5).select("customer_id")
        relations = self.data["relations"].join(filtered_users, on="customer_id", how="right")

        # Create a mapping from the original id to a new id
        users = filtered_users.with_columns(customer_id_map=pl.col("customer_id").cast(pl.Categorical).to_physical())
        articles = self.data["items"].with_columns(article_id_map=pl.col("article_id").cast(pl.String).cast(pl.Categorical).to_physical())

        users_id_map = users.select("customer_id", "customer_id_map").unique()
        articles_id_map = articles.select("article_id", "article_id_map").unique()

        # Add path column to articles
        article_path_tuple_list = [(int(i.stem), str(i)) for i in Path(".data/hm/base/images").rglob("*.jpg")]
        articles_path_map = pl.DataFrame(
            {"article_id": [i[0] for i in article_path_tuple_list], "path": [i[1] for i in article_path_tuple_list]}
        )
        articles = articles.join(articles_path_map, on="article_id", how="left")

        # Add mapping columns to relations
        relations = (
            relations.sort("t_dat").join(users_id_map, on="customer_id", how="left").join(articles_id_map, on="article_id", how="left")
        )

        self.data["users"] = users
        self.data["items"] = articles
        self.data["relations"] = relations

    def save_intermediate(self, data=None, prefix=None):
        if data is None:
            data = self.data

        if prefix is not None:
            self._prefix = prefix

        self.intermediate.mkdir(parents=True, exist_ok=True)

        data["users"].write_parquet(self.intermediate / "customers.parquet")
        data["items"].write_parquet(self.intermediate / "articles.parquet")
        data["relations"].write_parquet(self.intermediate / "transactions_train.parquet")

    def load(self):
        self.data = {
            "relations": pl.read_parquet(self.intermediate / "transactions_train.parquet"),
            "users": pl.read_parquet(self.intermediate / "customers.parquet"),
            "items": pl.read_parquet(self.intermediate / "articles.parquet"),
        }

    @property
    def namings(self):
        return {
            "user_id": "customer_id",
            "item_id": "article_id",
            "user_id_map": "customer_id_map",
            "item_id_map": "article_id_map",
            "date": "t_dat",
        }

    def build_sparsifier(self):
        return Sparsifier(self.data["relations"], self.data["users"], self.data["items"], namings=self.namings)

    @property
    def n_users(self) -> int:
        return self.data["users"].get_column("customer_id_map").max() + 1  # type: ignore

    @property
    def n_items(self) -> int:
        return self.data["items"].get_column("article_id_map").max() + 1  # type: ignore
