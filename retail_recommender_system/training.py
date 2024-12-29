import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt


@dataclass
class History:
    _log_filename: ClassVar[str] = "history_log.json"

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    metrics: dict[str, list[float]] = field(default_factory=lambda: {"train_roc_auc": [], "test_roc_auc": []})
    eval_metrics: dict[str, dict[int, list[float]]] = field(default_factory=lambda: {"precision": {}, "recall": {}})

    def log_loss(self, train_loss: float, val_loss: float):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

    def log_metrics(self, metrics: dict[str, float]):
        self.metrics["train_roc_auc"].append(metrics["train_roc_auc"])
        self.metrics["test_roc_auc"].append(metrics["test_roc_auc"])

    def log_eval_metrics(self, metrics: dict[str, dict[int, float]]):
        for metric in metrics:
            for k in metrics[metric]:
                if k not in self.eval_metrics[metric]:
                    self.eval_metrics[metric][k] = []
                self.eval_metrics[metric][k].append(metrics[metric][k])

    def save(self, path: Path) -> None:
        with (path / self._log_filename).open("w") as f:
            json.dump(self.__dict__, f)

    def plot(self, path: Path) -> None:
        nrows = 2 + len(self.eval_metrics)
        fig, ax = plt.subplots(nrows, 1, figsize=(5 * nrows, 10))

        ax[0].plot(self.train_loss, label="train loss")
        ax[0].plot(self.val_loss, label="val loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        for metric in self.metrics:
            ax[1].plot(self.metrics[metric], label=metric)
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Value")
        ax[1].legend()

        for i, metric in enumerate(self.eval_metrics):
            i += 2
            for k in self.eval_metrics[metric]:
                ax[i].plot(self.eval_metrics[metric][k], label=f"{metric}@{k}")
            ax[i].set_xlabel("Epoch")
            ax[i].set_ylabel("Value")
            ax[i].legend()

        plt.savefig(path / "history_plot.png")
