from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl


class BaseDataset(ABC):
    def __init__(self, base: str, prefix: str):
        self._base = Path(base)
        self._prefix = Path(prefix)
        self.data = self.load()

    @property
    @abstractmethod
    def ds(self) -> str: ...

    @property
    def _root(self) -> Path:
        return self._base / self.ds

    @property
    def base(self) -> Path:
        return self._root / "base"

    @property
    def intermediate(self) -> Path:
        return self._root / "intermediate" / self._prefix

    @abstractmethod
    def load(self) -> dict[str, pl.DataFrame]: ...
