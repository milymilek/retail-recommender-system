import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def load_model(
    cls: type[nn.Module], model_path: Path, model_kwargs: dict[str, Any], device: torch.device = torch.device("cpu")
) -> nn.Module:
    model = cls(**model_kwargs)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model.to(device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(True)
