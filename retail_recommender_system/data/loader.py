from dataclasses import dataclass
from enum import Enum
from typing import TypedDict

import polars as pl

from retail_recommender_system.data.datasets.base import BaseDataset
from retail_recommender_system.data.datasets.hm import HMDataset
from retail_recommender_system.data.datasets.steam import SteamDataset


def load_dataset(config: "DataConfig") -> BaseDataset:
    return config.dataset.value(base_input=config.base, base_output=config.base_output, prefix=config.prefix)


class DatasetEnum(Enum):
    hm = HMDataset
    steam = SteamDataset


@dataclass
class DataConfig:
    dataset: DatasetEnum
    prefix: str
    base: str = ".data"
    base_output: str | None = None

    def __post_init__(self):
        if isinstance(self.dataset, str):
            self.dataset = DatasetEnum[self.dataset]

        if self.base_output is None:
            self.base_output = self.base
