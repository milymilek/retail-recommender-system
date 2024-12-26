from functools import partial
from pathlib import Path
from typing import Any, Callable

import polars as pl
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from retail_recommender_system.evaluation.metrics import precision_k, recall_k
from retail_recommender_system.evaluation.prediction import recommend_k
from retail_recommender_system.logging import init_logger
from retail_recommender_system.models.mf import MF, MFDataset, MFEvalDataset, MFModelConfig, collate_fn, eval_collate_fn
from retail_recommender_system.trainer.base import BaseTrainer
from retail_recommender_system.training import History
from retail_recommender_system.utils import batch_dict_to_device, save_model

logger = init_logger(__name__)


class MFTrainer(BaseTrainer):
    @property
    def _model_config(self) -> type:
        return MFModelConfig

    def _init_model(self) -> nn.Module:
        return MF(self._model_config(n_users=self.dataset.n_users, n_items=self.dataset.n_items, **self.model_config.model_config))

    def _init_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.train_config.lr)

    def _init_criterion(self) -> nn.Module:
        return nn.BCEWithLogitsLoss()

    def _init_datasets(self) -> dict[str, Dataset]:
        X_train, X_valid, _, _ = train_test_split(
            self.dataset.data["relations"], torch.ones(len(self.dataset.data["relations"])).numpy(), test_size=self.train_config.valid_size
        )

        train_dataset = MFDataset(X_train, n_items=self.dataset.n_items, neg_sampl=self.train_config.neg_sampl)
        val_dataset = MFDataset(X_valid, n_items=self.dataset.n_items, neg_sampl=self.train_config.neg_sampl)
        eval_dataset = MFEvalDataset(val_dataset.users_set(), val_dataset.items_set(), user_batch_size=1024)

        return {"train": train_dataset, "val": val_dataset, "eval": eval_dataset}

    def _init_loaders(self) -> dict[str, DataLoader]:
        train_loader = DataLoader(self.datasets["train"], batch_size=self.train_config.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(self.datasets["val"], batch_size=self.train_config.batch_size, shuffle=False, collate_fn=collate_fn)
        eval_loader = DataLoader(self.datasets["eval"], batch_size=1, collate_fn=eval_collate_fn, shuffle=False)

        return {"train": train_loader, "val": val_loader, "eval": eval_loader}

    @torch.no_grad
    def recommend_udf(self, batch: dict[str, torch.Tensor], model: nn.Module, n_items: int) -> torch.Tensor:
        model.eval()
        return model(batch).view(-1, n_items)

    def evaluate(self, K: list[int]) -> dict[str, Any]:
        users_set = self.loaders["eval"].dataset.users_set
        items_set = self.loaders["eval"].dataset.items_set
        ground_truth = self.loaders["val"].dataset.ground_truth()
        n_users = self.dataset.n_users
        n_items = self.dataset.n_items

        recommendations = recommend_k(
            partial(self.recommend_udf, model=self.model, n_items=len(items_set)),
            self.loaders["eval"],
            max(K),
            past_interactions=None,
        )

        metrics = {"precision": {}, "recall": {}}
        for k in K:
            precision = precision_k(recommendations, ground_truth, k=k, users_idx=users_set, n_users=n_users, n_items=n_items)
            recall = recall_k(recommendations, ground_truth, k=k, users_idx=users_set, n_users=n_users, n_items=n_items)

            metrics["precision"][k] = precision.item()
            metrics["recall"][k] = recall.item()
        return metrics

    def train(self, device: torch.device, print_every: None | int = None) -> tuple[float, float]:
        self.model.train()
        train_loss = 0.0
        preds, ground_truths = [], []

        for batch_idx, batch in enumerate(self.loaders["train"]):
            data = batch_dict_to_device(batch, device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, data["target"])
            loss.backward()
            self.optimizer.step()

            loss_item = loss.detach().cpu().item()

            if print_every is not None and batch_idx % print_every == 0:
                print(
                    "Train (Batch): [{}/{} ({:.0f}%)]\tTrain Loss: {:.4f}".format(
                        batch_idx, len(self.loaders["train"]), 100.0 * batch_idx / len(self.loaders["train"]), loss_item
                    )  # type: ignore
                )

            preds.append(output)
            ground_truths.append(data["target"])
            train_loss += loss_item

        train_loss /= len(self.loaders["train"])

        pred = torch.cat(preds, dim=0).detach().sigmoid().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        train_roc_auc = roc_auc_score(ground_truth, pred)

        return train_loss, train_roc_auc

    def test(self, device: torch.device, print_every: None | int = None) -> tuple[float, float]:
        self.model.eval()
        test_loss = 0.0
        preds, ground_truths = [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.loaders["val"]):
                data = batch_dict_to_device(batch, device)

                output = self.model(data)
                loss = self.criterion(output, data["target"])

                preds.append(output)
                ground_truths.append(data["target"])
                test_loss += loss.detach().cpu().item()

        pred = torch.cat(preds, dim=0).sigmoid().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        test_roc_auc = roc_auc_score(ground_truth, pred)
        test_loss /= len(self.loaders["val"])

        if print_every is not None:
            print(
                "\nTest: Test loss: {:.4f}".format(test_loss)  # type: ignore
            )

        return test_loss, test_roc_auc

    def fit(self) -> History:
        device = torch.device("cpu")
        logger.info("Training model %s on device %s", self.model.__class__.__name__, device)
        history = History()

        for epoch in tqdm(range(1, self.train_config.epochs + 1)):
            train_loss, train_roc_auc = self.train(device=device, print_every=100)
            test_loss, test_roc_auc = self.test(device=device, print_every=1)
            eval_metrics = self.evaluate(K=[5, 10, 12])

            history.log_loss(train_loss, test_loss)
            history.log_metrics({"train_roc_auc": train_roc_auc, "test_roc_auc": test_roc_auc})
            history.log_eval_metrics(eval_metrics)

        return history
