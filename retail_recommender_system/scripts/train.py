import argparse
import json
from typing import Any, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from retail_recommender_system.data.loader import DataConfig, load_dataset
from retail_recommender_system.logging import init_logger
from retail_recommender_system.models.mf import MF, MFDataset, MFEvalDataset, collate_fn, eval_collate_fn
from retail_recommender_system.trainer.loader import ModelConfig, TrainConfig, load_trainer
from retail_recommender_system.utils import create_log_dir, save_model, set_seed

logger = init_logger(__name__)


def main(args: argparse.Namespace):
    set_seed(args.seed)
    device = torch.device(args.device)

    data_config = DataConfig(**args.config["data"])
    dataset = load_dataset(data_config)

    model_config = ModelConfig(**args.config["model"])
    train_config = TrainConfig(**args.config["train"])
    trainer = load_trainer(model_config, train_config, dataset)
    history = trainer.fit()

    log_dir = create_log_dir(trainer.model)
    if args.save_weights:
        save_model(trainer.model, log_dir / "weights.pth")
    history.save(log_dir)
    history.plot(log_dir)


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
    parser.add_argument("--save_weights", type=bool, default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
