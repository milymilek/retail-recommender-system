import polars as pl

from retail_recommender_system.data.datasets.base import BaseDataset


class SteamDataset(BaseDataset):
    @property
    def ds(self) -> str:
        return "steam"

    def load(self) -> dict[str, pl.DataFrame]:
        return {}

    def cardinality(self) -> tuple[int, int]:
        return 0, 0
