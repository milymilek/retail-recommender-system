import torch

from retail_recommender_system.evaluation.metrics import ap_k, map_k, precision_k, recall_k


def test_precision_k(recommendations: torch.Tensor, ground_truth: torch.Tensor, users_idx: torch.Tensor, n_users: int, n_items: int):
    expected_precisions = [0.0, 0.25, 1.0 / 3, 0.25]

    for k, xprecision in zip(range(1, 5), expected_precisions):
        precision = precision_k(recommendations, ground_truth, k, users_idx, n_users, n_items)
        assert torch.isclose(precision, torch.tensor(xprecision), atol=1e-6)


def test_recall_k(recommendations: torch.Tensor, ground_truth: torch.Tensor, users_idx: torch.Tensor, n_users: int, n_items: int):
    expected_recalls = [0.0, 0.5, 0.75, 0.75]

    for k, xrecall in zip(range(1, 5), expected_recalls):
        recall = recall_k(recommendations, ground_truth, k, users_idx, n_users, n_items)
        assert torch.isclose(recall, torch.tensor(xrecall), atol=1e-6)


def test_map_k(recommendations: torch.Tensor, ground_truth: torch.Tensor, users_idx: torch.Tensor, n_users: int, n_items: int):
    expected_mean_average_precisions = [0.0, 0.125, 7.0 / 36, 20.0 / 96]

    # ap = ap_k(recommendations, ground_truth, 4, users_idx, n_users, n_items)
    # print(ap)

    for k, xmap in zip(range(1, 5), expected_mean_average_precisions):
        map = map_k(recommendations, ground_truth, k, users_idx, n_users, n_items)
        print(map)
        assert torch.isclose(map, torch.tensor(xmap), atol=1e-6)
