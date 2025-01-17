from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from retail_recommender_system.evaluation.metrics.precision import precision_k
from retail_recommender_system.evaluation.metrics.recall import recall_k
from retail_recommender_system.evaluation.prediction import recommend_k
from retail_recommender_system.logging import init_logger
from retail_recommender_system.models.mfconv import (
    MFConv,
    MFConv2,
    MFConvDataset,
    MFConvEvalDataset,
    MFConvModelConfig,
    collate_fn,
    eval_collate_fn,
)
from retail_recommender_system.trainer.mf import MFTrainer
from retail_recommender_system.utils import filter_set, split_by_time

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
        X_train, X_valid = self.dataset.data["relations"]

        train_dataset = MFConvDataset(
            relations=X_train,
            users=self.dataset.data["users"],
            items=self.dataset.data["items"],
            images_path=Path(".data/hm/intermediate/sep_2020/images_tensor.pt"),
            namings=self.dataset.namings,
            neg_sampl=self.train_config.neg_sampl,
        )
        val_dataset = MFConvDataset(
            relations=X_valid,
            users=self.dataset.data["users"],
            items=self.dataset.data["items"],
            images_path=Path(".data/hm/intermediate/sep_2020/images_tensor.pt"),
            namings=self.dataset.namings,
            neg_sampl=self.train_config.neg_sampl,
        )
        eval_dataset = MFConvEvalDataset(
            base_dataset=val_dataset,
            user_batch_size=self.train_config.eval_user_batch_size,
        )

        return {"train": train_dataset, "val": val_dataset, "eval": eval_dataset}

    def _init_loaders(self) -> dict[str, DataLoader]:
        train_loader = DataLoader(self.datasets["train"], batch_size=self.train_config.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(self.datasets["val"], batch_size=self.train_config.batch_size, shuffle=False, collate_fn=collate_fn)
        eval_loader = DataLoader(
            self.datasets["eval"], batch_size=self.train_config.eval_batch_size, shuffle=False, collate_fn=eval_collate_fn, drop_last=False
        )

        return {"train": train_loader, "val": val_loader, "eval": eval_loader}

    @torch.no_grad
    def recommend_udf(self, batch: dict[str, torch.Tensor], model: nn.Module, n_items: int) -> torch.Tensor:
        model.eval()
        return model.recommend(batch)

    def evaluate(self, K: list[int]) -> dict[str, Any]:
        past_interactions = self.loaders["train"].dataset.ground_truth
        users_set = self.loaders["eval"].dataset.users_set
        items_set = self.loaders["eval"].dataset.items_set
        ground_truth = self.loaders["eval"].dataset.ground_truth
        n_users = self.dataset.n_users
        n_items = self.dataset.n_items

        self.model.precompute_img_embeddings(self.loaders["eval"].dataset.images.cpu().to(self.device))

        recommendations = recommend_k(
            partial(self.recommend_udf, model=self.model, n_items=len(items_set)),
            self.loaders["eval"],
            max(K),
            device=self.device,
            past_interactions=past_interactions,
            n_users=n_users,
            n_items=n_items,
        )

        metrics = {"precision": {}, "recall": {}}
        for k in K:
            precision = precision_k(recommendations, ground_truth, k=k, users_idx=users_set, n_users=n_users, n_items=n_items)
            recall = recall_k(recommendations, ground_truth, k=k, users_idx=users_set, n_users=n_users, n_items=n_items)

            metrics["precision"][k] = precision.item()
            metrics["recall"][k] = recall.item()

            print(f"Precision@{k}: {precision.item()} | Recall@{k}: {recall.item()}")
        return metrics


class MFConv2Trainer(MFTrainer):
    @property
    def _model_config(self) -> type:
        return MFConvModelConfig

    def _init_model(self) -> nn.Module:
        model = MFConv2(
            self._model_config(n_users=self.dataset.n_users, n_items=self.dataset.n_items, **self.model_config.model_config)
        ).to(self.device)
        logger.info("Model: %s", model)
        return model
