from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

import polars as pl

from retail_recommender_system.data.datasets.base import BaseDataset
from retail_recommender_system.data.datasets.hm import HMDataset
from retail_recommender_system.data.datasets.steam import SteamDataset


def load_dataset(config: "DataConfig") -> BaseDataset:
    return config.dataset.value(base=config.base, prefix=config.prefix)


class DatasetEnum(Enum):
    hm = HMDataset
    steam = SteamDataset


@dataclass
class DataConfig:
    dataset: DatasetEnum
    prefix: str
    base: str = ".data"

    def __post_init__(self):
        if isinstance(self.dataset, str):
            self.dataset = DatasetEnum[self.dataset]
