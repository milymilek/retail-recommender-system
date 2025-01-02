import argparse
import json
from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import torch

from retail_recommender_system.data.loader import DataConfig, DatasetEnum, load_dataset
from retail_recommender_system.logging import init_logger
from retail_recommender_system.trainer.loader import ModelConfig, TrainConfig, load_trainer
from retail_recommender_system.utils import create_log_dir, parse_enum, save_model, set_seed

logger = init_logger(__name__)


def main(args: argparse.Namespace):
    set_seed(args.seed)

    dataset = args.dataset(base=args.base, prefix=args.prefix)
    dataset.load()

    sparsifier = dataset.build_sparsifier()
    sparsified_data = sparsifier.sparsify(args.frac)

    dataset.save_intermediate(sparsified_data, prefix=f"frac_{str(args.frac).replace('.', '_')}")

    print(sparsified_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=partial(parse_enum, cls=DatasetEnum), required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--base", type=str, required=False, default=".data")
    parser.add_argument("--frac", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
