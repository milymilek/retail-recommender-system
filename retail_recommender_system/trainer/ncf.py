import torch.nn as nn

from retail_recommender_system.logging import init_logger
from retail_recommender_system.models.ncf import NCF, NCFModelConfig
from retail_recommender_system.trainer.mf import MFTrainer

logger = init_logger(__name__)


class NCFTrainer(MFTrainer):
    @property
    def _model_config(self) -> type:
        return NCFModelConfig

    def _init_model(self) -> nn.Module:
        return NCF(self._model_config(n_users=self.dataset.n_users, n_items=self.dataset.n_items, **self.model_config.model_config))
