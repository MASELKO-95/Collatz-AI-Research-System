"""
Unit tests for the native C++ engine bindings.

Tests cover:
- C++ library loading
- Function bindings
- Data generation via C++ engine
- Comparison with Python implementation
"""
import pytest
import numpy as np
import os
from pathlib import Path


# Try to import native_engine, skip tests if not available
try:
    from native_engine import (
        generate_batch_native,
        is_native_available,
        get_stopping_time_native,
    )
    NATIVE_AVAILABLE = is_native_available()
except ImportError:
    NATIVE_AVAILABLE = False


@pytest.mark.skipif(not NATIVE_AVAILABLE, reason="Native C++ engine not available")
class TestNativeEngine:
    """Test the native C++ engine bindings."""

    def test_native_available(self):
        """Test that native engine is available."""
        assert is_native_available() is True

    def test_generate_batch_native_shape(self):
        """Test that native batch generation produces correct shapes."""
        start = 10
        end = 20
        max_len = 100
        
        numbers, stopping_times, parity_vectors = generate_batch_native(
            start, end, max_len
        )
        
        batch_size = end - start
        assert len(numbers) == batch_size
        assert len(stopping_times) == batch_size
        assert parity_vectors.shape == (batch_size, max_len)

    def test_generate_batch_native_values(self):
        """Test that native batch contains correct values."""
        start = 100
        end = 110
        max_len = 100
        
        numbers, stopping_times, parity_vectors = generate_batch_native(
            start, end, max_len
        )
        
        # Check number range
        assert np.all(numbers >= start)
        assert np.all(numbers < end)
        
        # Check stopping times are positive
        assert np.all(stopping_times > 0)
        
        # Check parity vectors contain valid values
        unique_values = np.unique(parity_vectors)
        assert all(v in [-1, 0, 1] for v in unique_values)

    def test_get_stopping_time_native(self):
        """Test native stopping time calculation."""
        # Test known values
        known_times = {
            1: 0,
            2: 1,
            3: 7,
            4: 2,
            5: 5,
            27: 111,
        }
        
        for n, expected_time in known_times.items():
            actual_time = get_stopping_time_native(n)
            assert actual_time == expected_time, f"Failed for n={n}"

    def test_native_vs_python_consistency(self):
        """Test that native and Python implementations produce same results."""
        # Import Python implementation
        from engine import get_stopping_time, get_parity_vector
        
        test_numbers = [10, 27, 100, 1000]
        
        for n in test_numbers:
            # Compare stopping times
            native_time = get_stopping_time_native(n)
            python_time = get_stopping_time(n)
            assert native_time == python_time, f"Stopping time mismatch for n={n}"

    def test_native_batch_consistency(self):
        """Test that native batch generation is consistent."""
        start = 50
        end = 60
        max_len = 100
        
        # Generate two batches with same parameters
        nums1, times1, vecs1 = generate_batch_native(start, end, max_len)
        nums2, times2, vecs2 = generate_batch_native(start, end, max_len)
        
        np.testing.assert_array_equal(nums1, nums2)
        np.testing.assert_array_equal(times1, times2)
        np.testing.assert_array_equal(vecs1, vecs2)

    def test_native_large_numbers(self):
        """Test native engine with larger numbers."""
        start = 10000
        end = 10010
        max_len = 200
        
        numbers, stopping_times, parity_vectors = generate_batch_native(
            start, end, max_len
        )
        
        assert len(numbers) == 10
        assert np.all(stopping_times > 0)

    def test_library_path(self):
        """Test that C++ library file exists."""
        src_dir = Path(__file__).parent.parent / "src"
        lib_path = src_dir / "libcollatz.so"
        
        # Library should exist if native is available
        if is_native_available():
            assert lib_path.exists(), "libcollatz.so not found"


@pytest.mark.skipif(NATIVE_AVAILABLE, reason="Testing fallback when native unavailable")
class TestNativeFallback:
    """Test behavior when native engine is not available."""

    def test_fallback_to_python(self):
        """Test that system falls back to Python implementation."""
        # This test runs when native is NOT available
        # Just verify we can still import and use Python version
        from engine import generate_batch_data
        
        numbers, stopping_times, parity_vectors = generate_batch_data(
            10, 20, 100
        )
        
        assert len(numbers) == 10
        assert len(stopping_times) == 10
        assert parity_vectors.shape == (10, 100)
