import torch


def validate_metric_inputs(recommendations: torch.Tensor, ground_truth: torch.Tensor, k: int) -> None:
    assert recommendations.ndim == 2, "`recommendations` should be a 2D tensor"
    assert ground_truth.ndim == 2, "`ground_truth` should be a 2D tensor"
    assert ground_truth.shape[0] == 2, "`ground_truth` should be a tensor of shape (2, -1)"
    assert recommendations.dtype == torch.float32, "`recommendations` should be a tensor of dtype torch.int64"
    assert ground_truth.dtype == torch.float32, "`ground_truth` should be a tensor of dtype torch.int64"
