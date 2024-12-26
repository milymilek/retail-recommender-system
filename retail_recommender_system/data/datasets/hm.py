import polars as pl

from retail_recommender_system.data.datasets.base import BaseDataset


class HMDataset(BaseDataset):
    @property
    def ds(self) -> str:
        return "hm"

    def load(self) -> dict[str, pl.DataFrame]:
        return {
            "relations": pl.read_parquet(self.intermediate / "relations.parquet"),
            "users": pl.read_parquet(self.intermediate / "users.parquet"),
            "items": pl.read_parquet(self.intermediate / "articles.parquet"),
        }

    @property
    def n_users(self) -> int:
        return self.data["users"].get_column("customer_id_map").max() + 1  # type: ignore

    @property
    def n_items(self) -> int:
        return self.data["items"].get_column("article_id_map").max() + 1  # type: ignore
