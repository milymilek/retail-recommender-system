import argparse
import json
from typing import Any

import polars as pl
import torch

from retail_recommender_system.data.loader import DataConfig, load_dataset
from retail_recommender_system.logging import init_logger
from retail_recommender_system.trainer.loader import ModelConfig, TrainConfig, load_trainer
from retail_recommender_system.utils import create_log_dir, save_model, set_seed

logger = init_logger(__name__)


def main(args: argparse.Namespace):
    set_seed(args.seed)

    relations = pl.read_csv(".data/steam/base/recommendations.csv", try_parse_dates=True)
    users = pl.read_csv(".data/steam/base/users.csv")
    games = pl.read_csv(".data/steam/base/games.csv")

    relations = relations.filter(pl.col("is_recommended") == True)
    filtered_users = relations.group_by("user_id").agg(pl.len()).filter(pl.col("len") >= 5).select("user_id")
    relations = relations.join(filtered_users, on="user_id", how="right")
    relations = relations.sort(by="date")

    filtered_users = users
    users = users.sort("user_id")
    users = users.with_columns(customer_id_map=pl.arange(0, len(users), 1))

    filtered_apps = relations.select("app_id").unique()
    games = games.join(filtered_apps, on="app_id", how="right")
    games = games.sort("app_id")
    games = games.with_columns(article_id_map=pl.arange(0, len(games), 1))

    relations = relations.join(users.select("user_id", "customer_id_map"), on="user_id", how="left")
    relations = relations.join(games.select("app_id", "article_id_map"), on="app_id", how="left")

    print(relations)

    # Write files to parquet
    users.write_parquet(".data/steam/intermediate/full/users.parquet")
    games.write_parquet(".data/steam/intermediate/full/games.parquet")
    relations.write_parquet(".data/steam/intermediate/full/relations.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
