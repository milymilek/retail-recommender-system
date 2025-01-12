from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl


class BaseDataset(ABC):
    def __init__(self, base_input: str, prefix: str, base_output: str | None = None, ignore_ds: bool = False):
        self._base_input = Path(base_input)
        self._base_output = Path(base_output) if base_output is not None else self._base_input
        self._prefix = Path(prefix)
        self._ignore_ds = ignore_ds

    @property
    @abstractmethod
    def ds(self) -> str: ...

    def _root(self, input_: bool = True) -> Path:
        base = self._base_input if input_ else self._base_output

        if self._ignore_ds and input_:
            return base
        return base / self.ds

    @property
    def base_path(self) -> Path:
        if self._ignore_ds:
            return self._root()
        return self._root() / "base"

    @property
    def intermediate(self) -> Path:
        return self._root(input_=False) / "intermediate" / self._prefix

    @property
    @abstractmethod
    def n_users(self) -> int: ...

    @property
    @abstractmethod
    def n_items(self) -> int: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def load_base(self) -> None: ...

    @abstractmethod
    def process_base(self) -> None: ...

    @abstractmethod
    def save_intermediate(self, data=None, prefix=None): ...
