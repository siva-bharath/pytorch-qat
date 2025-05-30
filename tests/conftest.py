import pytest
import torch

@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42) 