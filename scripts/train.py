import argparse
import random
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from retail_recommender_system.models.mf import MF, MFDataset, collate_fn
from retail_recommender_system.utils import load_model, set_seed


def _batch_dict_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: v.to(device) for k, v in batch.items()}


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
        data = _batch_dict_to_device(batch, device)

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
            data = _batch_dict_to_device(batch, device)

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


def _load_data():
    relations = pl.read_parquet(".data/intermediate/relations.parquet")
    users = pl.read_parquet(".data/intermediate/users.parquet")
    items = pl.read_parquet(".data/intermediate/articles.parquet")

    return relations, users, items


def fit(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
):
    print(f"> Training model[{model.__class__.__name__}] on device[{device}] begins...")

    history = {"train_loss": [], "val_loss": []}

    for epoch in tqdm(range(1, epochs + 1)):
        train(model, criterion, optimizer, train_loader, device, epoch, print_every=100)
        test(model, criterion, device, val_loader, print_every=1)


def main(args: argparse.Namespace):
    args = get_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    relations, users, items = _load_data()

    X_train, X_valid, _, _ = train_test_split(relations, np.ones(len(relations)), test_size=0.2, random_state=args.seed)

    batch_size = 4096
    neg_sampl = 3
    n_users = users.get_column("customer_id_map").max()
    n_items = items.get_column("article_id_map").max()

    train_dataset = MFDataset(X_train, n_items=n_items, neg_sampl=neg_sampl)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    val_dataset = MFDataset(X_valid, n_items=n_items, neg_sampl=neg_sampl)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    embedding_size = 16
    lr = 1e-4

    model = MF(n_users, n_items, emb_size=embedding_size).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    fit(model, criterion, optimizer, train_loader, val_loader, device, args.epochs)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)
