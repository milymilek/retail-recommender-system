import torch

from retail_recommender_system.evaluation.evaluation import recommendation_relevance
from retail_recommender_system.evaluation.utils import validate_metric_inputs


def precision_k(
    recommendations: torch.Tensor, ground_truth: torch.Tensor, k: int, users_idx: torch.Tensor, n_users: int, n_items: int
) -> torch.Tensor:
    recommendations_k = recommendations[:, :k]
    rel, _, rel_mask = recommendation_relevance(recommendations_k, ground_truth, users_idx=users_idx, n_users=n_users, n_items=n_items)

    return torch.mean(torch.mean(rel[rel_mask], dim=1))
