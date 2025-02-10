import torch

from retail_recommender_system.evaluation.evaluation import recommendation_relevance


def ap_k(
    recommendations: torch.Tensor, ground_truth: torch.Tensor, k: int, users_idx: torch.Tensor, n_users: int, n_items: int
) -> torch.Tensor:
    p_k = []
    for _k in range(1, k + 1):
        recommendations_k = recommendations[:, :_k]
        rel, _, rel_mask = recommendation_relevance(recommendations_k, ground_truth, users_idx=users_idx, n_users=n_users, n_items=n_items)
        p_k.append(torch.mean(rel[rel_mask], dim=1))
    return torch.vstack(p_k).mean(dim=0).view(-1, 1)


def map_k(
    recommendations: torch.Tensor, ground_truth: torch.Tensor, k: int, users_idx: torch.Tensor, n_users: int, n_items: int
) -> torch.Tensor:
    return torch.mean(ap_k(recommendations, ground_truth, k, users_idx, n_users, n_items))
