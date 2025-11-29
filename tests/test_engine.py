"""
Unit tests for the Numba-optimized data generation engine.

Tests cover:
- Collatz sequence generation
- Stopping time calculation
- Parity vector generation
- Batch data generation
"""
import pytest
import numpy as np
from engine import (
    next_collatz,
    get_stopping_time,
    get_parity_vector,
    generate_batch_data,
    detect_cycle,
)


class TestNextCollatz:
    """Test the next_collatz function."""

    def test_even_number(self):
        """Test that even numbers are divided by 2."""
        assert next_collatz(10) == 5
        assert next_collatz(100) == 50
        assert next_collatz(2) == 1

    def test_odd_number(self):
        """Test that odd numbers follow 3n+1 rule."""
        assert next_collatz(3) == 10
        assert next_collatz(5) == 16
        assert next_collatz(7) == 22

    def test_one(self):
        """Test edge case of n=1."""
        # 1 is odd, so 3*1+1 = 4
        assert next_collatz(1) == 4


class TestGetStoppingTime:
    """Test the get_stopping_time function."""

    def test_known_stopping_times(self, known_stopping_times):
        """Test against known stopping times."""
        for n, expected_time in known_stopping_times.items():
            actual_time = get_stopping_time(n)
            assert actual_time == expected_time, f"Failed for n={n}"

    def test_power_of_two(self):
        """Test that powers of 2 have stopping time = log2(n)."""
        assert get_stopping_time(2) == 1
        assert get_stopping_time(4) == 2
        assert get_stopping_time(8) == 3
        assert get_stopping_time(16) == 4
        assert get_stopping_time(1024) == 10

    def test_large_number(self):
        """Test that large numbers don't cause overflow."""
        # 27 has a notably long stopping time
        stopping_time = get_stopping_time(27)
        assert stopping_time == 111

    def test_returns_positive(self, sample_numbers):
        """Test that stopping times are always non-negative."""
        for n in sample_numbers:
            if n > 0:
                assert get_stopping_time(n) >= 0


class TestGetParityVector:
    """Test the get_parity_vector function."""

    def test_parity_vector_length(self):
        """Test that parity vector has correct length."""
        max_len = 100
        vec, actual_len = get_parity_vector(27, max_len)
        assert len(vec) == max_len
        assert actual_len <= max_len

    def test_parity_vector_values(self):
        """Test that parity vector contains only 0, 1, or -1 (padding)."""
        vec, _ = get_parity_vector(27, 200)
        unique_values = np.unique(vec)
        assert all(v in [-1, 0, 1] for v in unique_values)

    def test_known_sequence(self):
        """Test parity vector for n=3."""
        # 3->10->5->16->8->4->2->1
        # Parities: 1(odd), 0(even), 1(odd), 0, 0, 0, 0
        vec, length = get_parity_vector(3, 100)
        expected_start = [1, 0, 1, 0, 0, 0, 0]
        assert list(vec[:length]) == expected_start

    def test_power_of_two_all_even(self):
        """Test that powers of 2 produce all-even parity vectors."""
        vec, length = get_parity_vector(16, 100)
        # 16->8->4->2->1, all divisions by 2 (even)
        assert all(vec[i] == 0 for i in range(length))

    def test_padding_is_negative_one(self):
        """Test that unused positions are padded with -1."""
        vec, length = get_parity_vector(2, 100)
        # After the sequence, should be -1
        assert all(vec[i] == -1 for i in range(length, 100))


class TestDetectCycle:
    """Test the detect_cycle function."""

    def test_converges_to_one(self, sample_numbers):
        """Test that normal numbers converge to 1."""
        for n in sample_numbers:
            if n > 0:
                result = detect_cycle(n, max_steps=10000)
                assert result == 0, f"Number {n} should converge to 1"

    def test_max_steps_limit(self):
        """Test that very large numbers respect max_steps."""
        # Use a number with long stopping time
        result = detect_cycle(27, max_steps=10)
        # Should either converge or hit limit
        assert result in [0, 2]


class TestGenerateBatchData:
    """Test the generate_batch_data function."""

    def test_batch_shape(self):
        """Test that batch data has correct shapes."""
        start, end = 10, 20
        max_len = 100
        numbers, stopping_times, parity_vectors = generate_batch_data(
            start, end, max_len
        )

        batch_size = end - start
        assert len(numbers) == batch_size
        assert len(stopping_times) == batch_size
        assert parity_vectors.shape == (batch_size, max_len)

    def test_batch_values(self):
        """Test that batch contains correct number range."""
        start, end = 100, 110
        numbers, _, _ = generate_batch_data(start, end, 100)
        
        assert np.all(numbers >= start)
        assert np.all(numbers < end)
        assert len(np.unique(numbers)) == end - start

    def test_stopping_times_positive(self):
        """Test that all stopping times are positive."""
        start, end = 50, 60
        _, stopping_times, _ = generate_batch_data(start, end, 100)
        
        assert np.all(stopping_times > 0)

    def test_parity_vectors_valid(self):
        """Test that parity vectors contain valid values."""
        start, end = 10, 15
        _, _, parity_vectors = generate_batch_data(start, end, 100)
        
        unique_values = np.unique(parity_vectors)
        assert all(v in [-1, 0, 1] for v in unique_values)

    def test_consistency(self):
        """Test that repeated calls produce same results for same input."""
        start, end = 20, 25
        max_len = 50
        
        numbers1, times1, vecs1 = generate_batch_data(start, end, max_len)
        numbers2, times2, vecs2 = generate_batch_data(start, end, max_len)
        
        np.testing.assert_array_equal(numbers1, numbers2)
        np.testing.assert_array_equal(times1, times2)
        np.testing.assert_array_equal(vecs1, vecs2)
