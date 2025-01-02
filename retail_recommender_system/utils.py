import random
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)


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
