import polars as pl

from retail_recommender_system.data.datasets.base import BaseDataset
from retail_recommender_system.data.sparsifier import Sparsifier


class SteamDataset(BaseDataset):
    @property
    def ds(self) -> str:
        return "steam"

    def load_base(self):
        self.data = {
            "relations": pl.read_csv(".data/steam/base/recommendations.csv", try_parse_dates=True),
            "users": pl.read_csv(".data/steam/base/users.csv"),
            "items": pl.read_csv(".data/steam/base/games.csv"),
        }

    def process_base(self):
        relations = self.data["relations"].filter(pl.col("is_recommended") == True)
        filtered_users = relations.group_by("user_id").agg(pl.len()).filter(pl.col("len") >= 5).select("user_id")
        relations = relations.join(filtered_users, on="user_id", how="right")
        relations = relations.sort(by="date")
        relations = relations.with_columns(target=pl.col("is_recommended").cast(pl.Int8))

        filtered_users = self.data["users"]
        users = self.data["users"].sort("user_id")
        users = users.with_columns(user_id_map=pl.arange(0, len(users), 1))

        filtered_apps = relations.select("app_id").unique()
        games = self.data["items"].join(filtered_apps, on="app_id", how="right")
        games = games.sort("app_id")
        games = games.with_columns(app_id_map=pl.arange(0, len(games), 1))

        relations = relations.join(users.select("user_id", "user_id_map"), on="user_id", how="left")
        relations = relations.join(games.select("app_id", "app_id_map"), on="app_id", how="left")

        self.data["users"] = users
        self.data["items"] = games
        self.data["relations"] = relations

    def save_intermediate(self, data=None, prefix=None):
        if data is None:
            data = self.data

        if prefix is not None:
            self._prefix = prefix

        self.intermediate.mkdir(parents=True, exist_ok=True)

        data["users"].write_parquet(self.intermediate / "users.parquet")
        data["items"].write_parquet(self.intermediate / "games.parquet")
        data["relations"].write_parquet(self.intermediate / "relations.parquet")

    def load(self):
        self.data = {
            "relations": pl.read_parquet(self.intermediate / "relations.parquet"),
            "users": pl.read_parquet(self.intermediate / "users.parquet"),
            "items": pl.read_parquet(self.intermediate / "games.parquet"),
        }

    @property
    def namings(self):
        return {"user_id": "user_id", "item_id": "app_id", "user_id_map": "user_id_map", "item_id_map": "app_id_map", "date": "date"}

    def build_sparsifier(self):
        return Sparsifier(self.data["relations"], self.data["users"], self.data["items"], namings=self.namings)

    @property
    def n_users(self) -> int:
        return self.data["users"].get_column("user_id_map").max() + 1  # type: ignore

    @property
    def n_items(self) -> int:
        return self.data["items"].get_column("app_id_map").max() + 1  # type: ignore
