from dataclasses import dataclass
from functools import cached_property
from typing import Any

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from torchvision.io import read_image

from retail_recommender_system.layers.dotprod import DotProd
from retail_recommender_system.logging import init_logger

logger = init_logger(__name__)


def collate_fn(batch):
    u_id = torch.cat([x["u_id"] for x in batch])
    i_id = torch.cat([x["i_id"] for x in batch])
    imgs = torch.cat([x["i_img"] for x in batch])
    target = torch.cat([x["target"] for x in batch])
    return {"u_id": u_id, "i_id": i_id, "i_img": imgs, "target": target}


def _to_float(x):
    return x.to(torch.float32)


def _normalize(x):
    return x / 255.0


def _to_rgb(x):
    return x.expand(3, -1, -1)


from collections import OrderedDict


class CacheDict(OrderedDict):
    """Dict with a limited length, ejecting LRUs as needed."""

    def __init__(self, *args, cache_len: int = 10, **kwargs):
        assert cache_len > 0
        self.cache_len = cache_len

        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().move_to_end(key)

        while len(self) > self.cache_len:
            oldkey = next(iter(self))
            super().__delitem__(oldkey)

    def __getitem__(self, key):
        val = super().__getitem__(key)
        super().move_to_end(key)

        return val


class MFConvDataset(Dataset):
    def __init__(self, relations: pl.DataFrame, users: pl.DataFrame, items: pl.DataFrame, namings, neg_sampl: int = 5):
        self._df = torch.from_numpy(relations.select(namings["user_id_map"], namings["item_id_map"]).to_numpy()).to(torch.int32)
        self._users = torch.from_numpy(users.get_column(namings["user_id_map"]).to_numpy()).to(torch.float32)
        self._items = items.get_column("path")
        self._neg_sampl = neg_sampl

        self.img_size = 28
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                _to_rgb,
                _to_float,
                _normalize,
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.img_cache = CacheDict(cache_len=20000)

    @property
    def _n_users(self) -> int:
        return len(self._users)

    @property
    def _n_items(self) -> int:
        return len(self._items)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._df[idx]
        user = row[0].unsqueeze(0)
        items = row[1].unsqueeze(0)

        u_id = user.repeat(self._neg_sampl + 1)
        i_id = torch.cat([items, self._approx_neg_sampl()])

        imgs = self.read_imgs(i_id)

        target = torch.tensor([1.0] + [0.0] * self._neg_sampl, dtype=torch.float)

        return {"u_id": u_id, "i_id": i_id, "i_img": imgs, "target": target}

    def _approx_neg_sampl(self):
        neg_i_id = torch.randint(low=0, high=self._n_items, size=(self._neg_sampl,), dtype=torch.int32)
        return neg_i_id

    def read_imgs(self, i_id: torch.Tensor):
        _imgs = []

        for i in i_id.tolist():
            img = self.img_cache.get(i)
            if img is None:
                path = self._items[int(i)]
                if path is None:
                    img = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float)
                else:
                    img = self.transform(read_image(path))

                self.img_cache[i] = img
            _imgs.append(img)
        return torch.stack(_imgs, dim=0)

    def users_set(self) -> torch.Tensor:
        return torch.arange(self._n_users, dtype=torch.int32)

    def items_set(self) -> torch.Tensor:
        return torch.arange(self._n_items, dtype=torch.int32)

    def ground_truth(self) -> torch.Tensor:
        return self._df.T


class MFConvEvalDataset(IterableDataset):
    def __init__(self, base_dataset: MFConvDataset, user_batch_size: int):
        super().__init__()
        self._base_dataset = base_dataset
        self._user_batch_size = user_batch_size

        # self._imgs = self._base_dataset.read_imgs(self.items_set)

    @cached_property
    def users_set(self) -> torch.Tensor:
        return self._base_dataset.users_set()

    @cached_property
    def items_set(self) -> torch.Tensor:
        return self._base_dataset.items_set()

    @cached_property
    def ground_truth(self) -> torch.Tensor:
        return self._base_dataset.ground_truth()

    def get_batch_data(self, batch):
        u_id = torch.repeat_interleave(batch, self._base_dataset._n_items)
        i_id = self.items_set.repeat(len(batch))

        print(self._imgs.shape)
        raise

        return {
            "u_id": u_id,
            "i_id": i_id,
            "i_img": self._imgs,
            "target": torch.zeros(len(u_id), dtype=torch.float),
        }

    def __len__(self):
        return len(self.users_set) // self._user_batch_size + 1

    def __iter__(self):
        for batch in self.users_set.split(self._user_batch_size):
            yield self.get_batch_data(batch)


@dataclass
class MFConvModelConfig:
    n_users: int
    n_items: int
    emb_size: int
    dropout: float = 0.0


class _ImgConv(nn.Module):
    def __init__(self, config: MFConvModelConfig):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=(3, 3), stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.ff_e = nn.Linear(300, config.emb_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        return self.ff_e(x)


class MFConv(nn.Module):
    def __init__(self, config: MFConvModelConfig):
        super().__init__()
        self.user_factors = nn.Embedding(config.n_users, config.emb_size)
        self.item_factors = nn.Embedding(config.n_items, config.emb_size)

        self.dropout = nn.Dropout(config.dropout)

        self.conv = _ImgConv(config)

        self.dot = DotProd()

    def forward(self, x):
        user_factors = self.dropout(self.user_factors(x["u_id"]))
        item_factors = self.dropout(self.item_factors(x["i_id"])) * self.conv(x["i_img"])
        return self.dot(user_factors, item_factors)
