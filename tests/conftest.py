"""
Pytest configuration and shared fixtures for Collatz AI tests.
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def device():
    """Provide CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_numbers():
    """Provide sample numbers for testing Collatz sequences."""
    return [1, 2, 3, 5, 7, 27, 100, 1000, 10000]


@pytest.fixture
def known_stopping_times():
    """Provide known stopping times for verification."""
    return {
        1: 0,
        2: 1,
        3: 7,
        4: 2,
        5: 5,
        6: 8,
        7: 16,
        8: 3,
        27: 111,
    }


@pytest.fixture
def sample_parity_vectors():
    """Provide sample parity vectors for testing."""
    # For n=3: 3->10->5->16->8->4->2->1
    # Parity: odd, even, odd, even, even, even, even
    # Binary: 1, 0, 1, 0, 0, 0, 0
    return {
        3: [1, 0, 1, 0, 0, 0, 0],
        5: [1, 0, 0, 0, 1],  # 5->16->8->4->2->1
    }


@pytest.fixture
def mock_model_config():
    """Provide mock configuration for model testing."""
    return {
        "d_model": 64,
        "nhead": 2,
        "num_layers": 2,
        "max_len": 100,
    }


@pytest.fixture
def sample_batch():
    """Provide a sample batch for testing."""
    batch_size = 4
    seq_len = 10
    
    # Create sample parity vectors (0s and 1s)
    parity_vectors = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.long)
    
    # Create sample stopping times
    stopping_times = torch.randint(5, 50, (batch_size,), dtype=torch.float32)
    
    # Create padding mask (all valid for this simple case)
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    return {
        "parity_vectors": parity_vectors,
        "stopping_times": stopping_times,
        "padding_mask": padding_mask,
    }
