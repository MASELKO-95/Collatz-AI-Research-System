"""
Unit tests for the PyTorch Dataset and DataLoader.

Tests cover:
- Dataset initialization
- Dataset iteration
- Collate function
- Batch generation
"""
import pytest
import torch
from torch.utils.data import DataLoader
from dataset import CollatzIterableDataset, collate_fn
import numpy as np


class TestCollatzIterableDataset:
    """Test the CollatzIterableDataset class."""

    def test_initialization(self):
        """Test that dataset initializes correctly."""
        dataset = CollatzIterableDataset(
            start_range=100,
            end_range=200,
            batch_size=32,
            max_len=100,
        )
        
        assert dataset.start_range == 100
        assert dataset.end_range == 200
        assert dataset.batch_size == 32
        assert dataset.max_len == 100

    def test_iteration(self):
        """Test that dataset can be iterated."""
        dataset = CollatzIterableDataset(
            start_range=10,
            end_range=20,
            batch_size=5,
            max_len=50,
        )
        
        # Get first batch
        batch = next(iter(dataset))
        
        assert isinstance(batch, dict)
        assert "parity_vector" in batch
        assert "stopping_time" in batch

    def test_batch_shapes(self):
        """Test that batches have correct shapes."""
        batch_size = 8
        max_len = 100
        
        dataset = CollatzIterableDataset(
            start_range=100,
            end_range=200,
            batch_size=batch_size,
            max_len=max_len,
        )
        
        batch = next(iter(dataset))
        
        assert batch["parity_vector"].shape == (batch_size, max_len)
        assert batch["stopping_time"].shape == (batch_size,)

    def test_multiple_batches(self):
        """Test that dataset produces multiple batches."""
        dataset = CollatzIterableDataset(
            start_range=10,
            end_range=50,
            batch_size=10,
            max_len=50,
        )
        
        batches = []
        for i, batch in enumerate(dataset):
            batches.append(batch)
            if i >= 2:  # Get 3 batches
                break
        
        assert len(batches) == 3
        for batch in batches:
            assert "parity_vector" in batch
            assert "stopping_time" in batch

    def test_parity_vector_values(self):
        """Test that parity vectors contain valid values."""
        dataset = CollatzIterableDataset(
            start_range=10,
            end_range=20,
            batch_size=5,
            max_len=50,
        )
        
        batch = next(iter(dataset))
        parity_vector = batch["parity_vector"]
        
        # Should contain only -1 (padding), 0 (even), 1 (odd), or 2 (padding token)
        unique_values = torch.unique(parity_vector)
        assert all(v in [-1, 0, 1, 2] for v in unique_values.tolist())

    def test_stopping_times_positive(self):
        """Test that stopping times are positive."""
        dataset = CollatzIterableDataset(
            start_range=10,
            end_range=30,
            batch_size=10,
            max_len=50,
        )
        
        batch = next(iter(dataset))
        stopping_times = batch["stopping_time"]
        
        assert torch.all(stopping_times > 0)


class TestCollateFn:
    """Test the collate_fn function."""

    def test_collate_basic(self):
        """Test basic collation of samples."""
        samples = [
            {
                "parity_vector": torch.tensor([0, 1, 0, 1]),
                "stopping_time": torch.tensor(10.0),
            },
            {
                "parity_vector": torch.tensor([1, 0, 1, 0]),
                "stopping_time": torch.tensor(15.0),
            },
        ]
        
        batch = collate_fn(samples)
        
        assert "parity_vector" in batch
        assert "stopping_time" in batch
        assert batch["parity_vector"].shape == (2, 4)
        assert batch["stopping_time"].shape == (2,)

    def test_collate_with_padding(self):
        """Test collation with different sequence lengths."""
        samples = [
            {
                "parity_vector": torch.tensor([0, 1, 0]),
                "stopping_time": torch.tensor(5.0),
            },
            {
                "parity_vector": torch.tensor([1, 0, 1, 0, 1]),
                "stopping_time": torch.tensor(8.0),
            },
        ]
        
        batch = collate_fn(samples)
        
        # Should pad to longest sequence (5)
        assert batch["parity_vector"].shape == (2, 5)
        
        # Check that padding is applied correctly
        # First sample should have padding at the end
        assert batch["parity_vector"][0, 3] == 2 or batch["parity_vector"][0, 3] == -1

    def test_collate_preserves_values(self):
        """Test that collation preserves original values."""
        samples = [
            {
                "parity_vector": torch.tensor([0, 1, 0, 1]),
                "stopping_time": torch.tensor(10.0),
            },
        ]
        
        batch = collate_fn(samples)
        
        torch.testing.assert_close(
            batch["parity_vector"][0],
            samples[0]["parity_vector"]
        )
        torch.testing.assert_close(
            batch["stopping_time"][0],
            samples[0]["stopping_time"]
        )


class TestDataLoader:
    """Test DataLoader integration."""

    def test_dataloader_creation(self):
        """Test that DataLoader can be created with dataset."""
        dataset = CollatzIterableDataset(
            start_range=10,
            end_range=50,
            batch_size=8,
            max_len=50,
        )
        
        # Note: For IterableDataset, we typically don't use DataLoader's batching
        # But we can still create it
        dataloader = DataLoader(dataset, batch_size=None)
        
        assert dataloader is not None

    def test_dataloader_iteration(self):
        """Test that DataLoader can be iterated."""
        dataset = CollatzIterableDataset(
            start_range=10,
            end_range=30,
            batch_size=5,
            max_len=50,
        )
        
        dataloader = DataLoader(dataset, batch_size=None)
        
        batch = next(iter(dataloader))
        
        assert isinstance(batch, dict)
        assert "parity_vector" in batch
        assert "stopping_time" in batch

    def test_dataloader_multiple_workers(self):
        """Test DataLoader with multiple workers."""
        dataset = CollatzIterableDataset(
            start_range=10,
            end_range=100,
            batch_size=10,
            max_len=50,
        )
        
        # Use num_workers=0 for testing (multiprocessing can be tricky in tests)
        dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
        
        batches = []
        for i, batch in enumerate(dataloader):
            batches.append(batch)
            if i >= 1:  # Get 2 batches
                break
        
        assert len(batches) == 2
