import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from retail_recommender_system.logging import init_logger
from retail_recommender_system.models.deepfm import DeepFM
from retail_recommender_system.models.mf import MF
from retail_recommender_system.models.mfconv import MFConv
from retail_recommender_system.models.ncf import NCF
from retail_recommender_system.trainer.deepfm import DeepFMTrainer
from retail_recommender_system.trainer.mf import MFTrainer
from retail_recommender_system.trainer.mfconv import MFConvTrainer
from retail_recommender_system.trainer.ncf import NCFTrainer

if TYPE_CHECKING:
    import torch

    from retail_recommender_system.data.datasets.base import BaseDataset
    from retail_recommender_system.trainer.base import BaseTrainer

logger = init_logger(__name__)


class ModelEnum(str, Enum):
    MF = MF
    MFConv = MFConv
    DeepFM = DeepFM
    NCF = NCF


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
    train_print_every: int
    eval_batch_size: int
    eval_user_batch_size: int
    eval_print_every: int
    neg_sampl: int
    lr: float
    epochs: int


def load_trainer(model_config: ModelConfig, train_config: TrainConfig, dataset: "BaseDataset", device: "torch.device") -> "BaseTrainer":
    logger.info("Model configuration:\n%s", json.dumps(asdict(model_config), indent=2))
    logger.info("Train configuration:\n%s", json.dumps(asdict(train_config), indent=2))

    trainer_map = {ModelEnum.MF: MFTrainer, ModelEnum.DeepFM: DeepFMTrainer, ModelEnum.NCF: NCFTrainer, ModelEnum.MFConv: MFConvTrainer}
    return trainer_map[model_config.model_type](model_config, train_config, dataset, device)
