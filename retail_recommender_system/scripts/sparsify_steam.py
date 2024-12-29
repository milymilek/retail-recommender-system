import argparse
import json
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import torch

from retail_recommender_system.data.loader import DataConfig, load_dataset
from retail_recommender_system.logging import init_logger
from retail_recommender_system.trainer.loader import ModelConfig, TrainConfig, load_trainer
from retail_recommender_system.utils import create_log_dir, save_model, set_seed

logger = init_logger(__name__)


def main(args: argparse.Namespace):
    set_seed(args.seed)

    relations = pl.read_parquet(".data/steam/intermediate/full/relations.parquet")
    users = pl.read_parquet(".data/steam/intermediate/full/users.parquet")
    items = pl.read_parquet(".data/steam/intermediate/full/games.parquet")

    n_users = relations.get_column("customer_id_map").n_unique()
    n_items = items.get_column("app_id").n_unique()
    relations_n_users = relations.get_column("customer_id").n_unique()
    relations_n_items = relations.get_column("app_id").n_unique()

    relations_filtered = relations.join(
        relations.select("customer_id_map").unique().sample(fraction=args.frac), on="customer_id_map", how="right"
    )
    relations_filtered_n_users = relations_filtered.get_column("customer_id").n_unique()
    relations_filtered_n_items = relations_filtered.get_column("app_id").n_unique()

    print(f"""(Users) Nunique: {n_users}
(Items) Nunique: {n_items}
(Relations, Users) Nunique: {relations_n_users} | diff: {n_users - relations_n_users}
(Relations, Items) Nunique: {relations_n_items} | diff: {n_items - relations_n_items}
(Relations, Users) Nunique (filtered): {relations_filtered_n_users} | diff: {n_users - relations_filtered_n_users}
(Relations, Items) Nunique (filtered): {relations_filtered_n_items} | diff: {n_items - relations_filtered_n_items}""")

    relations_filtered.shape
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    relations_cnt = relations.group_by("customer_id_map").agg(pl.len()).sort("len", descending=True)
    relations_filtered_cnt = relations_filtered.group_by("customer_id_map").agg(pl.len()).sort("len", descending=True)
    ax[0].hist(relations_cnt.select("len"), bins=500)
    ax[1].hist(relations_filtered_cnt.select("len"), bins=500)
    users_filtered = users.join(relations_filtered.select("customer_id").unique(), on="customer_id", how="inner").drop("customer_id_map")
    items_filtered = items.join(relations_filtered.select("app_id").unique(), on="app_id", how="inner")
    users_filtered = users_filtered.with_columns(customer_id_map=pl.col("customer_id").cast(pl.Categorical).to_physical())
    items_filtered = items_filtered.with_columns(article_id_map=pl.col("app_id").cast(pl.String).cast(pl.Categorical).to_physical())

    users_id_map = users_filtered.select("customer_id", "customer_id_map").unique()
    articles_id_map = items_filtered.select("app_id", "article_id_map").unique()
    for c, id_map in zip(["customer_id", "article_id"], [users_id_map, articles_id_map]):
        id_map.write_parquet(f".data/hm/intermediate/frac_{str(args.frac).replace('.', '_')}/{c}_map.parquet")
    relations_filtered = (
        relations_filtered.drop("customer_id_map", "article_id_map")
        .sort("t_dat")
        .join(users_id_map, on="customer_id", how="left")
        .join(articles_id_map, on="article_id", how="left")
    )
    # Write files to parquet
    users_filtered.write_parquet(f".data/steam/intermediate/frac_{str(args.frac).replace('.', '_')}/users.parquet")
    items_filtered.write_parquet(f".data/steam/intermediate/frac_{str(args.frac).replace('.', '_')}/games.parquet")
    relations_filtered.write_parquet(f".data/steam/intermediate/frac_{str(args.frac).replace('.', '_')}/relations.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frac", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
