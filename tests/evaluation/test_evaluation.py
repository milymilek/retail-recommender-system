import torch

from retail_recommender_system.evaluation.evaluation import recommendation_relevance


def test_recommendation_relevance(
    recommendations: torch.Tensor, ground_truth: torch.Tensor, users_idx: torch.Tensor, n_users: int, n_items: int
):
    rel, rel_sum, rel_mask = recommendation_relevance(recommendations, ground_truth, users_idx, n_users, n_items)

    assert rel.shape == recommendations.shape
    assert torch.equal(rel_sum, torch.tensor([1, 2, 0]))
    assert torch.equal(rel_mask, torch.tensor([True, True, False]))
