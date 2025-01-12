from functools import partial
from typing import Any

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from retail_recommender_system.evaluation.metrics import precision_k, recall_k
from retail_recommender_system.evaluation.prediction import recommend_k
from retail_recommender_system.logging import init_logger
from retail_recommender_system.models.mf import MF, MFDataset, MFEvalDataset, MFModelConfig, collate_fn, eval_collate_fn
from retail_recommender_system.trainer.base import BaseTrainer
from retail_recommender_system.training import History
from retail_recommender_system.utils import batch_dict_to_device

logger = init_logger(__name__)


class MFTrainer(BaseTrainer):
    @property
    def _model_config(self) -> type:
        return MFModelConfig

    def _init_model(self) -> nn.Module:
        model = MF(self._model_config(n_users=self.dataset.n_users, n_items=self.dataset.n_items, **self.model_config.model_config)).to(
            self.device
        )
        logger.info("Model: %s", model)
        return model

    def _init_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.train_config.lr)

    def _init_criterion(self) -> nn.Module:
        return nn.BCEWithLogitsLoss()

    def _init_scheduler(self) -> Any:
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.5)

    def _init_datasets(self) -> dict[str, Dataset]:
        X_train, X_valid = self.dataset.data["relations"]

        train_dataset = MFDataset(
            relations=X_train,
            users=self.dataset.data["users"],
            items=self.dataset.data["items"],
            namings=self.dataset.namings,
            neg_sampl=self.train_config.neg_sampl,
        )
        val_dataset = MFDataset(
            relations=X_valid,
            users=self.dataset.data["users"],
            items=self.dataset.data["items"],
            namings=self.dataset.namings,
            neg_sampl=self.train_config.neg_sampl,
        )
        eval_dataset = MFEvalDataset(
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

    def train(self, print_every: None | int = None) -> tuple[float, float]:
        self.model.train()
        train_loss = 0.0
        preds, ground_truths = [], []

        for batch_idx, batch in enumerate(self.loaders["train"]):
            data = batch_dict_to_device(batch, self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, data["target"])
            loss.backward()
            self.optimizer.step()

            loss_item = loss.detach().cpu().item()

            if print_every is not None and batch_idx % print_every == 0:
                percentage = 100.0 * batch_idx / len(self.loaders["train"])
                print(f"Train (Batch): [{batch_idx}/{len(self.loaders['train'])} ({percentage:.0f}%)] | Loss: {loss_item:.4f}")

            preds.append(output)
            ground_truths.append(data["target"])
            train_loss += loss_item

        train_loss /= len(self.loaders["train"])

        pred = torch.cat(preds, dim=0).detach().sigmoid().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).detach().cpu().numpy()
        train_roc_auc = float(roc_auc_score(ground_truth, pred))

        print(f"\nTrain: Loss: {train_loss:.4f} | ROC AUC: {train_roc_auc:.4f}")

        return train_loss, train_roc_auc

    def test(self) -> tuple[float, float]:
        self.model.eval()
        test_loss = 0.0
        preds, ground_truths = [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.loaders["val"]):
                data = batch_dict_to_device(batch, self.device)

                output = self.model(data)
                loss = self.criterion(output, data["target"])

                preds.append(output)
                ground_truths.append(data["target"])
                test_loss += loss.detach().cpu().item()

        pred = torch.cat(preds, dim=0).sigmoid().cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        test_roc_auc = float(roc_auc_score(ground_truth, pred))
        test_loss /= len(self.loaders["val"])

        print(f"Test: Loss: {test_loss:.4f} | ROC AUC: {test_roc_auc:.4f}")

        return test_loss, test_roc_auc

    def fit(self) -> History:
        logger.info("Training model %s on device %s", self.model.__class__.__name__, self.device)
        history = History()

        for epoch in tqdm(range(1, self.train_config.epochs + 1)):
            train_loss, train_roc_auc = self.train(print_every=self.train_config.train_print_every)
            test_loss, test_roc_auc = self.test()

            history.log_loss(train_loss, test_loss)
            history.log_metrics({"train_roc_auc": train_roc_auc, "test_roc_auc": test_roc_auc})

            if epoch % self.train_config.eval_print_every == 0:
                eval_metrics = self.evaluate(K=[5, 10, 12])
                history.log_eval_metrics(eval_metrics)

            self.scheduler.step()
            print("LR: ", self.scheduler.get_last_lr())

        return history
