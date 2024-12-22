import torch

from retail_recommender_system.evaluation.evaluation import recommendation_relevance
from retail_recommender_system.evaluation.utils import validate_metric_inputs


def precision_k(recommendations: torch.Tensor, ground_truth: torch.Tensor, k: int, n_items: int) -> torch.Tensor:
    # validate_metric_inputs(recommendations, ground_truth, k)

    recommendations_k = recommendations[:, :k]
    rel, _, rel_mask = recommendation_relevance(recommendations_k, ground_truth, n_items)

    return torch.mean(torch.mean(rel[rel_mask], dim=1))
