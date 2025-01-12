from typing import TYPE_CHECKING, Callable

import torch
from scipy.sparse import csr_array
from tqdm import tqdm

from retail_recommender_system.utils import batch_dict_to_device

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

_default_ignored_interaction_value: int = -1


def _remove_past_interactions(prob: torch.Tensor, batch: dict[str, torch.Tensor], past_interactions_csr: csr_array) -> None:
    user_batch = batch["u_id"].unique()
    past_interactions_coords = past_interactions_csr[user_batch].tocoo().coords
    prob[past_interactions_coords] = _default_ignored_interaction_value


def recommend_k(
    recommend_udf: Callable,
    loader: "DataLoader",
    k: int,
    device: torch.device,
    past_interactions: torch.Tensor | None = None,
    n_users: int | None = None,
    n_items: int | None = None,
) -> torch.Tensor:
    if past_interactions is not None:
        past_interactions_csr = csr_array(
            (torch.ones_like(past_interactions[0]), (past_interactions[0], past_interactions[1])), shape=(n_users, n_items)
        )
        assert past_interactions.shape[0] == 2, "`past_interactions` should be a tensor of shape (2, -1)"

    recommend_batches = []
    for batch in tqdm(loader):
        batch_device = batch_dict_to_device(batch, device)
        prob = recommend_udf(batch_device).cpu()

        if past_interactions is not None:
            _remove_past_interactions(prob, batch, past_interactions_csr)

        recommend_batches.append(prob.topk(k, dim=1).indices)

    return torch.cat(recommend_batches, 0).to(torch.int32)
