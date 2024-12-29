import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, sizes: list[int], batch_norm: bool = False, dropout: float | None = None) -> None:
        super(FeedForward, self).__init__()

        layers = []
        for in_dim, out_dim in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(in_dim, out_dim))

            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))

            layers.append(nn.ReLU())

            if dropout is not None:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(sizes[-1], 1))

        self._layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers(x)
