import torch
from scipy.sparse import csr_array


def recommendation_relevance(
    recommendations: torch.Tensor, ground_truth: torch.Tensor, users_idx: torch.Tensor, n_users: int, n_items: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert ground_truth.shape[0] == 2, "`ground_truth` should be a tensor of shape (2, -1)"
    ground_truth_csr = csr_array((torch.ones_like(ground_truth[0]), (ground_truth[0], ground_truth[1])), shape=(n_users, n_items))

    k = recommendations.shape[1]
    user_idx = users_idx.repeat_interleave(k).tolist()
    item_idx = recommendations.flatten().tolist()
    relevance = torch.from_numpy(ground_truth_csr[user_idx, item_idx]).view((len(recommendations), k)).to(torch.float32)
    relevance_sum = torch.from_numpy(ground_truth_csr.sum(axis=1)).to(torch.float32)[users_idx]
    relevance_mask = relevance_sum > 0

    return relevance, relevance_sum, relevance_mask
