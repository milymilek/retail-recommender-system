import torch
from scipy.sparse import csr_array
from torch.utils.data import IterableDataset


def recommendation_relevance(
    recommendations: torch.Tensor, ground_truth: torch.Tensor, n_items: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_users = recommendations.shape[0]

    assert ground_truth.shape[0] == 2, "`ground_truth` should be a tensor of shape (2, -1)"
    ground_truth_csr = csr_array((torch.ones_like(ground_truth[0]), (ground_truth[0], ground_truth[1])), shape=(n_users, n_items))

    k = recommendations.shape[1]
    user_idx = torch.arange(n_users).repeat_interleave(k).tolist()
    item_idx = recommendations.flatten().tolist()
    relevance = torch.from_numpy(ground_truth_csr[user_idx, item_idx]).view((n_users, k)).to(torch.float32)
    relevance_sum = torch.from_numpy(ground_truth_csr.sum(axis=1)).to(torch.float32)
    relevance_mask = relevance_sum > 0

    return relevance, relevance_sum, relevance_mask


class EvalDataset(IterableDataset):
    def __init__(self, n_users, n_items, user_batch_size):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items

        self.user_batch_size = user_batch_size

    def get_batch_data(self, batch):
        u_min, u_max = batch, min(batch + self.user_batch_size, self.n_users)
        u_id = torch.repeat_interleave(torch.arange(u_min, u_max), self.n_items)
        i_id = torch.arange(self.n_items).repeat(u_max - u_min)

        return torch.column_stack((u_id, i_id))

    def __len__(self):
        return self.n_users // self.user_batch_size + 1

    def __iter__(self):
        for batch in range(0, self.n_users, self.user_batch_size):
            yield self.get_batch_data(batch)
