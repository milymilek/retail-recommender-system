import pytest
import torch


@pytest.fixture
def recommendations() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 1, 2, 3],  # User 0 recommendations
            [4, 5, 6, 7],  # User 1 recommendations
            [0, 2, 4, 6],  # User 3 recommendations
        ]
    )


@pytest.fixture
def ground_truth() -> torch.Tensor:
    return torch.tensor(
        [
            [0, 1, 1],
            [1, 6, 2],
            # Users 2 and 3 have no interactions
        ]
    )


@pytest.fixture
def users_idx() -> torch.Tensor:
    return torch.tensor([0, 1, 3])  # Ignore user 2


@pytest.fixture
def n_users(recommendations: torch.Tensor) -> int:
    return 4


@pytest.fixture
def n_items(recommendations: torch.Tensor) -> int:
    return 8  # type: ignore
