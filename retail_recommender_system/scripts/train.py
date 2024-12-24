import argparse
import json
import logging
from typing import Any, TypedDict

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from retail_recommender_system.data.loader import DataConfig, load_dataset
from retail_recommender_system.evaluation.metrics import precision_k, recall_k
from retail_recommender_system.evaluation.prediction import recommend_k
from retail_recommender_system.logging import init_logger
from retail_recommender_system.models.mf import MF, MFDataset, MFEvalDataset, collate_fn, eval_collate_fn, fit, recommend_udf
from retail_recommender_system.utils import create_log_dir, load_model, plot_epoch_metrics, save_model, set_seed

logger = init_logger(__name__)


class TrainConfig(TypedDict):
    valid_size: float
    batch_size: int
    neg_sampl: int
    lr: float
    epochs: int


class MFModelConfig(TypedDict):
    n_users: int
    n_items: int
    emb_size: int


def main(args: argparse.Namespace):
    set_seed(args.seed)
    device = torch.device(args.device)

    data_config = DataConfig(**args.config["data"])

    dataset = load_dataset(data_config)
    n_users, n_items = dataset.cardinality()

    train_config = TrainConfig(**args.config["train"])
    logger.info("Train configuration:\n%s", json.dumps(train_config, indent=2))

    X_train, X_valid, _, _ = train_test_split(
        dataset.data["relations"], np.ones(len(dataset.data["relations"])), test_size=train_config["valid_size"], random_state=args.seed
    )
    users_set = torch.from_numpy(X_valid.select("customer_id_map").unique().sort(by="customer_id_map").to_numpy().flatten()).to(torch.int32)
    items_set = torch.from_numpy(X_valid.select("article_id_map").unique().sort(by="article_id_map").to_numpy().flatten()).to(torch.int32)
    ground_truth = torch.from_numpy((X_valid.select("customer_id_map", "article_id_map").to_numpy())).to(torch.int32).T

    train_dataset = MFDataset(X_train, n_items=n_items, neg_sampl=train_config["neg_sampl"])
    train_loader = DataLoader(train_dataset, batch_size=train_config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_dataset = MFDataset(X_valid, n_items=n_items, neg_sampl=train_config["neg_sampl"])
    val_loader = DataLoader(val_dataset, batch_size=train_config["batch_size"], shuffle=False, collate_fn=collate_fn)
    eval_dataset = MFEvalDataset(users_set, items_set, user_batch_size=1024)
    eval_loader = DataLoader(eval_dataset, batch_size=1, collate_fn=eval_collate_fn, shuffle=False)

    model_config = MFModelConfig(n_users=n_users, n_items=n_items, **args.config["model"])
    logger.info("Model configuration:\n%s", json.dumps(model_config, indent=2))
    model = MF(**model_config).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_config["lr"])

    history = fit(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        eval_loader,
        ground_truth,
        users_set,
        items_set,
        n_users,
        n_items,
        device,
        train_config["epochs"],
    )
    log_dir = create_log_dir(model)
    save_model(model, log_dir / "weights.pth")
    history.save(log_dir)


def parse_json(file_path: str) -> dict[str, Any]:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise argparse.ArgumentTypeError(f"Config file '{file_path}' not found.")
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError(f"Config file '{file_path}' is not a valid JSON file.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=parse_json, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
