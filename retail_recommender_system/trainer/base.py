from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.nn.Module
    import torch.optim.Optimizer
    from torch.utils.data import DataLoader, Dataset

    from retail_recommender_system.data.datasets.base import BaseDataset
    from retail_recommender_system.trainer.loader import ModelConfig, TrainConfig
    from retail_recommender_system.training import History


class BaseTrainer(ABC):
    def __init__(self, model_config: "ModelConfig", train_config: "TrainConfig", dataset: "BaseDataset"):
        self.model_config = model_config
        self.train_config = train_config
        self.dataset = dataset
        self.model = self._init_model()
        self.optimizer = self._init_optimizer()
        self.criterion = self._init_criterion()
        self.datasets = self._init_datasets()
        self.loaders = self._init_loaders()

    @property
    @abstractmethod
    def _model_config(self) -> type: ...

    @abstractmethod
    def _init_model(self) -> "torch.nn.Module": ...

    @abstractmethod
    def _init_optimizer(
        self,
    ) -> "torch.optim.Optimizer": ...

    @abstractmethod
    def _init_criterion(self) -> "torch.nn.Module": ...

    @abstractmethod
    def _init_datasets(self) -> dict[str, "Dataset"]: ...

    @abstractmethod
    def _init_loaders(self) -> dict[str, "DataLoader"]: ...

    @abstractmethod
    def fit() -> "History": ...
