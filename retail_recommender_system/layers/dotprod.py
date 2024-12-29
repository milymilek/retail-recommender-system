import torch
import torch.nn as nn


class DotProd(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sum(x * y, dim=1)
