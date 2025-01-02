import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from retail_recommender_system.logging import init_logger
from retail_recommender_system.models.mfconv import MFConv, MFConvDataset, MFConvEvalDataset, MFConvModelConfig, collate_fn
from retail_recommender_system.trainer.mf import MFTrainer
from retail_recommender_system.utils import split_by_time

logger = init_logger(__name__)


class MFConvTrainer(MFTrainer):
    @property
    def _model_config(self) -> type:
        return MFConvModelConfig

    def _init_model(self) -> nn.Module:
        model = MFConv(self._model_config(n_users=self.dataset.n_users, n_items=self.dataset.n_items, **self.model_config.model_config)).to(
            self.device
        )
        logger.info("Model: %s", model)
        return model

    def _init_datasets(self) -> dict[str, Dataset]:
        X_train, X_valid = split_by_time(
            self.dataset.data["relations"], date_col=self.dataset.namings["date"], validation_ratio=self.train_config.valid_size
        )

        train_dataset = MFConvDataset(
            relations=X_train,
            users=self.dataset.data["users"],
            items=self.dataset.data["items"],
            namings=self.dataset.namings,
            neg_sampl=self.train_config.neg_sampl,
        )
        val_dataset = MFConvDataset(
            relations=X_valid,
            users=self.dataset.data["users"],
            items=self.dataset.data["items"],
            namings=self.dataset.namings,
            neg_sampl=self.train_config.neg_sampl,
        )
        eval_dataset = MFConvEvalDataset(base_dataset=val_dataset, user_batch_size=self.train_config.eval_user_batch_size)

        return {"train": train_dataset, "val": val_dataset, "eval": eval_dataset}

    def _init_loaders(self) -> dict[str, DataLoader]:
        train_loader = DataLoader(
            self.datasets["train"], batch_size=self.train_config.batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn
        )
        val_loader = DataLoader(self.datasets["val"], batch_size=self.train_config.batch_size, shuffle=False, collate_fn=collate_fn)
        eval_loader = DataLoader(self.datasets["eval"], batch_size=self.train_config.eval_batch_size, collate_fn=collate_fn, shuffle=False)

        return {"train": train_loader, "val": val_loader, "eval": eval_loader}
