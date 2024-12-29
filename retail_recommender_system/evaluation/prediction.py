from typing import TYPE_CHECKING, Callable

import torch
from tqdm import tqdm

from retail_recommender_system.utils import batch_dict_to_device

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

_default_ignored_interaction_value: int = -1


def _remove_past_interactions(prob: torch.Tensor, batch: dict[str, torch.Tensor], past_interactions: torch.Tensor):
    prob[past_interactions[0], past_interactions[1]] = _default_ignored_interaction_value


def recommend_k(
    recommend_udf: Callable, loader: "DataLoader", k: int, device: torch.device, past_interactions: torch.Tensor | None = None
) -> torch.Tensor:
    if past_interactions is not None:
        assert past_interactions.shape[0] == 2, "`past_interactions` should be a tensor of shape (2, -1)"

    recommend_batches = []
    for batch in tqdm(loader):
        batch = batch_dict_to_device(batch, device)

        prob = recommend_udf(batch).cpu()

        if past_interactions is not None:
            _remove_past_interactions(prob, batch, past_interactions)

        recommend_batches.append(prob.topk(k, 1)[1])

    return torch.cat(recommend_batches, 0).to(torch.int32)
