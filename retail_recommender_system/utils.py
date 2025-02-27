import random
from collections.abc import Callable
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision.io import read_image
from tqdm import tqdm

from retail_recommender_system.logging import init_logger

logger = init_logger(__name__)


def load_model(
    cls: type[nn.Module], model_path: Path, model_kwargs: dict[str, Any], device: torch.device = torch.device("cpu")
) -> nn.Module:
    logger.info("Loading model %s from %s", cls, model_path)
    model = cls(**model_kwargs)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model.to(device)


def save_model(model: nn.Module, model_path: Path) -> None:
    logger.info("Saving model %s to %s", type(model), model_path)
    torch.save(model.state_dict(), model_path)


def set_seed(seed: int) -> None:
    logger.info("Setting seed to %d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_epoch_metrics(**kwargs) -> None:
    fig, ax = plt.subplots(1, len(kwargs), figsize=(12, 5))

    for i, (title, data) in enumerate(kwargs.items()):
        ax[i].plot(data)
        ax[i].set_xlabel("Epochs")
        ax[i].set_title(title)

    plt.legend()
    plt.show()


def batch_dict_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) for k, v in batch.items()}


def create_log_dir(model: nn.Module, base: str = ".runs") -> Path:
    path = Path(f"{base}/{model.__class__.__name__}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_enum(x: str, cls: type[Enum]):
    return cls[x].value


def split_by_time(df: pl.DataFrame, date_col: str, validation_ratio: float = 0.2) -> tuple[pl.DataFrame, pl.DataFrame]:
    df = df.sort(date_col)
    split_idx = int(len(df) * (1 - validation_ratio))
    return df[:split_idx], df[split_idx:]


def filter_set(df: pl.DataFrame, reference_df: pl.DataFrame, user_col: str, item_col: str) -> pl.DataFrame:
    users = reference_df[user_col].unique().to_numpy().ravel()
    items = reference_df[item_col].unique().to_numpy().ravel()

    return df.filter(pl.col(user_col).is_in(users), pl.col(item_col).is_in(items))


def read_imgs(paths: list[str], transform: Callable, default_size: tuple[int, int, int], tqdm_: bool = False) -> torch.Tensor:
    _imgs = []

    iterator = paths
    if tqdm_:
        iterator = tqdm(iterator)

    for path in iterator:
        if path is None:
            img = torch.zeros(default_size, dtype=torch.float)
        else:
            img = transform(read_image(path))
        _imgs.append(img)
    return torch.stack(_imgs, dim=0)


def approx_neg_sampl(n_items: int, neg_sampl: int) -> torch.Tensor:
    return torch.randint(low=0, high=n_items, size=(neg_sampl,), dtype=torch.int32)


def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {non_trainable_params}")

    return {"total_params": total_params, "trainable_params": trainable_params, "non_trainable_params": non_trainable_params}


def plot_accuracy_per_class(train_accuracy_per_class, valid_accuracy_per_class, weights):
    fig, ax = plt.subplots(2, 2, figsize=(20, 4))

    sns.heatmap(
        [np.array(train_accuracy_per_class)],
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar=True,
        cbar_kws={"label": "Accuracy"},
        annot_kws={"rotation": 90, "va": "center"},
        ax=ax[0, 0],
        vmin=0,
        vmax=1,
    )
    ax[0, 0].set_xlabel("Class Index")
    ax[0, 0].set_title("Train Accuracy per Class (Heatmap)")
    ax[0, 0].set_yticks([])
    ax[0, 0].set_xticks(np.arange(len(train_accuracy_per_class)) + 0.5)
    ax[0, 0].set_xticklabels(np.arange(len(train_accuracy_per_class)), rotation=90)

    sns.heatmap(
        [np.array(valid_accuracy_per_class)],
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar=True,
        cbar_kws={"label": "Accuracy"},
        annot_kws={"rotation": 90, "va": "center"},
        ax=ax[1, 0],
        vmin=0,
        vmax=1,
    )
    ax[1, 0].set_xlabel("Class Index")
    ax[1, 0].set_title("Valid Accuracy per Class (Heatmap)")
    ax[1, 0].set_yticks([])
    ax[1, 0].set_xticks(np.arange(len(valid_accuracy_per_class)) + 0.5)
    ax[1, 0].set_xticklabels(np.arange(len(valid_accuracy_per_class)), rotation=90)

    assert np.isclose(np.sum(weights), 1), f"Weights must sum to 1, got {np.sum(weights)}"
    weighted_train_accuracy = train_accuracy_per_class * weights
    sns.heatmap(
        [weighted_train_accuracy],
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar=True,
        cbar_kws={"label": "Weighted Accuracy"},
        annot_kws={"rotation": 90, "va": "center"},
        ax=ax[0, 1],
        vmin=0,
        vmax=1,
    )
    ax[0, 1].set_xlabel("Class Index")
    ax[0, 1].set_title("Weighted Train Accuracy per Class (Heatmap)")
    ax[0, 1].set_yticks([])
    ax[0, 1].set_xticks(np.arange(len(weighted_train_accuracy)) + 0.5)
    ax[0, 1].set_xticklabels(np.arange(len(weighted_train_accuracy)), rotation=90)

    weighted_valid_accuracy = valid_accuracy_per_class * weights
    sns.heatmap(
        [weighted_valid_accuracy],
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        cbar=True,
        cbar_kws={"label": "Weighted Accuracy"},
        annot_kws={"rotation": 90, "va": "center"},
        ax=ax[1, 1],
        vmin=0,
        vmax=1,
    )
    ax[1, 1].set_xlabel("Class Index")
    ax[1, 1].set_title("Weighted Valid Accuracy per Class (Heatmap)")
    ax[1, 1].set_yticks([])
    ax[1, 1].set_xticks(np.arange(len(weighted_valid_accuracy)) + 0.5)
    ax[1, 1].set_xticklabels(np.arange(len(weighted_valid_accuracy)), rotation=90)

    plt.tight_layout()
    plt.show()
