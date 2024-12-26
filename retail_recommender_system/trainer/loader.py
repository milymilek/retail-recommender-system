import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from retail_recommender_system.data.datasets.base import BaseDataset
from retail_recommender_system.logging import init_logger
from retail_recommender_system.models.deepfm import DeepFM, DeepFMModelConfig
from retail_recommender_system.models.mf import MF, MFModelConfig
from retail_recommender_system.trainer.base import BaseTrainer
from retail_recommender_system.trainer.deepfm import DeepFMTrainer
from retail_recommender_system.trainer.mf import MFTrainer

logger = init_logger(__name__)


class ModelEnum(str, Enum):
    MF = MF
    DeepFM = DeepFM


@dataclass
class ModelConfig:
    model_type: ModelEnum
    model_config: dict[str, Any]

    def __post_init__(self):
        if isinstance(self.model_type, str):
            self.model_type = ModelEnum[self.model_type]


@dataclass
class TrainConfig:
    valid_size: float
    batch_size: int
    neg_sampl: int
    lr: float
    epochs: int


def load_trainer(model_config: ModelConfig, train_config: TrainConfig, dataset: BaseDataset) -> BaseTrainer:
    logger.info("Model configuration:\n%s", json.dumps(asdict(model_config), indent=2))
    logger.info("Train configuration:\n%s", json.dumps(asdict(train_config), indent=2))

    trainer_map = {ModelEnum.MF: MFTrainer, ModelEnum.DeepFM: DeepFMTrainer}
    return trainer_map[model_config.model_type](model_config, train_config, dataset)
