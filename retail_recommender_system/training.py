import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass
class History:
    _log_filename: ClassVar[str] = "history_log.json"

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    eval_metrics: dict[str, dict[int, list[float]]] = field(default_factory=lambda: {"precision": {}, "recall": {}})

    def log_loss(self, train_loss: float, val_loss: float):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

    def log_eval_metrics(self, metrics: dict[str, dict[int, float]]):
        for metric in metrics:
            for k in metrics[metric]:
                if k not in self.eval_metrics[metric]:
                    self.eval_metrics[metric][k] = []
                self.eval_metrics[metric][k].append(metrics[metric][k])

    def save(self, path: Path) -> None:
        with (path / self._log_filename).open("w") as f:
            json.dump(self.__dict__, f)
