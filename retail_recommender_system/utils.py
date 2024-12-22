from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def load_model(
    cls: type[nn.Module], model_path: Path, model_kwargs: dict[str, Any], device: torch.device = torch.device("cpu")
) -> nn.Module:
    model = cls(**model_kwargs)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model.to(device)
