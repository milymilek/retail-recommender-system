from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset

from retail_recommender_system.layers.dotprod import DotProd
from retail_recommender_system.logging import init_logger
from retail_recommender_system.utils import approx_neg_sampl, count_model_parameters

logger = init_logger(__name__)


def collate_fn(batch):
    u_id = torch.cat([x["u_id"] for x in batch])
    i_id = torch.cat([x["i_id"] for x in batch])
    imgs = torch.cat([x["i_img"] for x in batch])
    target = torch.cat([x["target"] for x in batch])
    return {"u_id": u_id, "i_id": i_id, "i_img": imgs, "target": target}


def eval_collate_fn(batch):
    u_id = torch.cat([x["u_id"] for x in batch])
    return {"u_id": u_id}


class MFConvDataset(Dataset):
    def __init__(
        self,
        relations: pl.DataFrame,
        users: pl.DataFrame,
        items: pl.DataFrame,
        images_path: Path,
        namings: dict[str, str],
        neg_sampl: int = 5,
    ):
        self._df = torch.from_numpy(relations.select(namings["user_id_map"], namings["item_id_map"]).to_numpy()).to(torch.int32)
        self._users = torch.from_numpy(users.get_column(namings["user_id_map"]).to_numpy()).to(torch.float32)
        self._items = items.get_column("path")
        self._neg_sampl = neg_sampl
        self._images = torch.load(images_path, weights_only=True)

    @property
    def _n_users(self) -> int:
        return len(self._users)

    @property
    def _n_items(self) -> int:
        return len(self._items)

    @property
    def images(self) -> torch.Tensor:
        return self._images

    @cached_property
    def users_set(self) -> torch.Tensor:
        return torch.arange(self._n_users, dtype=torch.int32)

    @cached_property
    def items_set(self) -> torch.Tensor:
        return torch.arange(self._n_items, dtype=torch.int32)

    @cached_property
    def ground_truth(self) -> torch.Tensor:
        return self._df.T

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._df[idx]
        user = row[0].unsqueeze(0)
        items = row[1].unsqueeze(0)

        u_id = user.repeat(self._neg_sampl + 1)
        i_id = torch.cat([items, approx_neg_sampl(self._n_items, self._neg_sampl)])

        imgs = self._images[i_id]

        target = torch.tensor([1.0] + [0.0] * self._neg_sampl, dtype=torch.float)

        return {"u_id": u_id, "i_id": i_id, "i_img": imgs, "target": target}


class MFConvEvalDataset(IterableDataset):
    def __init__(self, base_dataset: MFConvDataset, user_batch_size: int):
        super().__init__()
        self._base_dataset = base_dataset
        self._user_batch_size = user_batch_size

    @property
    def users_set(self) -> torch.Tensor:
        return self._base_dataset.users_set

    @property
    def items_set(self) -> torch.Tensor:
        return self._base_dataset.items_set

    @property
    def ground_truth(self) -> torch.Tensor:
        return self._base_dataset.ground_truth

    @property
    def images(self) -> torch.Tensor:
        return self._base_dataset.images

    def __len__(self):
        return len(self.users_set) // self._user_batch_size + 1

    def __iter__(self):
        for batch in self.users_set.split(self._user_batch_size):
            yield {"u_id": batch}


@dataclass
class MFConvModelConfig:
    n_users: int
    n_items: int
    emb_size: int
    image_size: tuple[int, int, int] | None = None
    dropout: float = 0.0
    pretrained_conv: str | None = None


class ImgConv(nn.Module):
    def __init__(self, config: MFConvModelConfig):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(3, 3), stride=2)
        self.conv2 = nn.Conv2d(6, 10, kernel_size=(3, 3), stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=1)
        self.ff_1 = nn.Linear(8410, 1024)
        self.ff_e = nn.Linear(1024, config.emb_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.dropout(self.maxpool(F.relu(self.conv1(x))))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.ff_1(x)))
        return self.ff_e(x)


class ImgConvLarge(nn.Module):
    def __init__(self, config: MFConvModelConfig):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=(4, 4), stride=3)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=(3, 3), stride=3)
        self.conv3 = nn.Conv2d(24, 36, kernel_size=(3, 3), stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=1)
        self.ff_e = nn.Linear(576, config.emb_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.dropout(self.maxpool(F.relu(self.conv1(x))))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.ff_e(x)


class AlexNet(nn.Module):
    def __init__(self, config: MFConvModelConfig):
        super().__init__()

        convs = [
            (
                nn.Conv2d(chs[0], chs[1], kernel_size=(k, k), stride=s),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            )
            for (chs, k, s) in [((3, 96), 9, 3), ((96, 256), 4, 1), ((256, 384), 3, 1), ((384, 256), 2, 1)]
        ]
        convs = [l for ls in convs for l in ls][:12]
        self.convs = nn.Sequential(*convs)
        self.mlps = nn.Sequential(
            nn.Linear(1536, 1024),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(512, config.emb_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)
        return self.mlps(x)


class MFConv(nn.Module):
    def __init__(self, config: MFConvModelConfig):
        super().__init__()
        self.user_factors = nn.Embedding(config.n_users, config.emb_size)
        self.item_factors = nn.Embedding(config.n_items, config.emb_size)

        self.dropout = nn.Dropout(config.dropout)

        self.conv = ImgConv(config)
        if config.pretrained_conv is not None:
            self._load_conv_weights(config.pretrained_conv)

        self.dot = DotProd()

    def _load_conv_weights(self, path: str):
        self.conv.load_state_dict(torch.load(path, weights_only=True))

        for name, param in self.conv.named_parameters():
            if name.startswith("ff_e"):
                param.requires_grad = True  # Keep ff_e trainable
            else:
                param.requires_grad = False  # Freeze all other layers

        print("Pretrained weights loaded. Layer `conv`'s Parameters: \n", count_model_parameters(self.conv))

    def forward(self, x: dict[str, torch.Tensor]):
        user_factors = self.dropout(self.user_factors(x["u_id"]))
        item_factors = self.dropout(self.item_factors(x["i_id"])) + self.conv(x["i_img"])
        return self.dot(user_factors, item_factors)

    @torch.no_grad()
    def precompute_img_embeddings(self, x):
        self._precomputed_img_emb = self.conv(x)

    @torch.no_grad()
    def recommend(self, x: dict[str, torch.Tensor]):
        user_emb = self.user_factors.weight[x["u_id"]]
        item_emb = self.item_factors.weight + self._precomputed_img_emb
        return torch.sigmoid(user_emb @ item_emb.T)


class MFConv2(nn.Module):
    def __init__(self, config: MFConvModelConfig):
        super().__init__()
        self.config = config
        self.num_classes = 36

        self.user_factors = nn.Embedding(config.n_users, config.emb_size)
        self.item_factors = nn.Embedding(config.n_items, config.emb_size)
        self._load_pretrained_item_factors()

        self.dropout = nn.Dropout(config.dropout)

        self.dot = DotProd()

    def _load_pretrained_item_factors(self):
        pretrained_weights = torch.load(".data/hm/intermediate/sep_2020/images_embeddings.pt")
        self.item_factors.weight = nn.Parameter(pretrained_weights)

    def forward(self, x: dict[str, torch.Tensor]):
        user_factors = self.dropout(self.user_factors(x["u_id"]))
        item_factors = self.dropout(self.item_factors(x["i_id"]))
        return self.dot(user_factors, item_factors)

    @torch.no_grad()
    def precompute_img_embeddings(self, x):
        self._precomputed_img_emb = self.conv

    @torch.no_grad()
    def recommend(self, x: dict[str, torch.Tensor]):
        user_emb = self.user_factors.weight[x["u_id"]]
        item_emb = self.item_factors.weight
        return torch.sigmoid(user_emb @ item_emb.T)
