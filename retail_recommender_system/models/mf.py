from functools import partial
from pathlib import Path
from typing import Any, Callable

import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from retail_recommender_system.evaluation.metrics import precision_k, recall_k
from retail_recommender_system.evaluation.prediction import recommend_k
from retail_recommender_system.logging import init_logger
from retail_recommender_system.training import History
from retail_recommender_system.utils import batch_dict_to_device, save_model

logger = init_logger(__name__)


@torch.no_grad
def recommend_udf(batch: dict[str, torch.Tensor], model: nn.Module, n_items: int) -> torch.Tensor:
    model.eval()
    return model(batch).view(-1, n_items)


def collate_fn(batch):
    u_id = torch.cat([x["u_id"] for x in batch])
    i_id = torch.cat([x["i_id"] for x in batch])
    target = torch.cat([x["target"] for x in batch])
    return {"u_id": u_id, "i_id": i_id, "target": target}


def eval_collate_fn(batch):
    batch = torch.cat(batch)
    return {"u_id": batch[:, 0], "i_id": batch[:, 1]}


def train(
    model: nn.Module,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    device: torch.device,
    epoch: int,
    print_every: None | int = None,
) -> float:
    model.train()
    train_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        data = batch_dict_to_device(batch, device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data["target"])
        loss.backward()
        optimizer.step()

        loss_item = loss.detach().cpu().item()

        if print_every is not None and batch_idx % print_every == 0:
            print(
                "Train (Batch): [{}/{} ({:.0f}%)]\tTrain Loss: {:.4f}".format(
                    batch_idx, len(train_loader), 100.0 * batch_idx / len(train_loader), loss_item
                )  # type: ignore
            )
        train_loss += loss_item

    train_loss /= len(train_loader)

    return train_loss


def test(model: nn.Module, loss_fn: Callable, device: torch.device, test_loader: DataLoader, print_every: None | int = None) -> float:
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            data = batch_dict_to_device(batch, device)

            output = model(data)
            loss = loss_fn(output, data["target"])

            loss_item = loss.detach().cpu().item()
            test_loss += loss_item

    test_loss /= len(test_loader)

    if print_every is not None:
        print(
            "\nTest: Test loss: {:.4f}".format(test_loss)  # type: ignore
        )

    return test_loss


def evaluate(
    model: nn.Module,
    eval_loader: DataLoader,
    ground_truth: torch.Tensor,
    users_set: torch.Tensor,
    items_set: torch.Tensor,
    n_users: int,
    n_items: int,
    K: list[int],
) -> dict[str, Any]:
    recommendations = recommend_k(partial(recommend_udf, model=model, n_items=len(items_set)), eval_loader, max(K), past_interactions=None)

    metrics = {"precision": {}, "recall": {}}
    for k in K:
        precision = precision_k(recommendations, ground_truth, k=k, users_idx=users_set, n_users=n_users, n_items=n_items)
        recall = recall_k(recommendations, ground_truth, k=k, users_idx=users_set, n_users=n_users, n_items=n_items)

        metrics["precision"][k] = precision.item()
        metrics["recall"][k] = recall.item()
    return metrics


def fit(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    eval_loader: DataLoader,
    ground_truth: torch.Tensor,
    users_set: torch.Tensor,
    items_set: torch.Tensor,
    n_users: int,
    n_items: int,
    device: torch.device,
    epochs: int,
) -> History:
    logger.info("Training model %s on device %s", model.__class__.__name__, device)
    history = History()

    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = train(model, criterion, optimizer, train_loader, device, epoch, print_every=100)
        test_loss = test(model, criterion, device, val_loader, print_every=1)
        eval_metrics = evaluate(model, eval_loader, ground_truth, users_set, items_set, n_users, n_items, K=[5, 10, 12])

        history.log_loss(train_loss, test_loss)
        history.log_eval_metrics(eval_metrics)

    return history


class MFDataset(Dataset):
    def __init__(self, df: pl.DataFrame, n_items: int, neg_sampl: int = 5):
        self._df = df
        self._n_items = n_items
        self._neg_sampl = neg_sampl

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._df[idx]
        user = torch.tensor(row.get_column("customer_id_map").to_numpy(), dtype=torch.int32)
        items = torch.tensor(row.get_column("article_id_map").to_numpy(), dtype=torch.int32)

        u_id = user.repeat(self._neg_sampl + 1)
        i_id = torch.cat([items, self._approx_neg_sampl()])
        target = torch.tensor([1.0] + [0.0] * self._neg_sampl, dtype=torch.float)

        return {"u_id": u_id, "i_id": i_id, "target": target}

    def _approx_neg_sampl(self):
        neg_i_id = torch.randint(low=0, high=self._n_items, size=(self._neg_sampl,), dtype=torch.int32)
        return neg_i_id


class MFEvalDataset(IterableDataset):
    def __init__(self, users_set: torch.Tensor, items_set: torch.Tensor, user_batch_size: int):
        super().__init__()
        self._users_set = users_set
        self._items_set = items_set

        self._user_batch_size = user_batch_size

    @property
    def _n_items(self):
        return self._items_set.shape[0]

    def get_batch_data(self, batch):
        u_id = torch.repeat_interleave(batch, self._n_items)
        i_id = self._items_set.repeat(len(batch))

        return torch.column_stack((u_id, i_id))

    def __len__(self):
        return len(self._users_set) // self._user_batch_size + 1

    def __iter__(self):
        for batch in self._users_set.split(self._user_batch_size):
            yield self.get_batch_data(batch)


class MF(nn.Module):
    def __init__(self, n_users, n_items, emb_size):
        super().__init__()
        self.user_factors = nn.Embedding(n_users, emb_size)
        self.item_factors = nn.Embedding(n_items, emb_size)

    def forward(self, x):
        user_factors = self.user_factors(x["u_id"])
        item_factors = self.item_factors(x["i_id"])
        return (user_factors * item_factors).sum(1)
