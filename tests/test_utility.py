"""
Unit tests for timekeep.utility
"""
import pytest

from timekeep.utility import *


class TestUtility:
    def test_uniform_length_is_true_if_no_empty_values(self):
        data = np.ones((4, 10, 2))
        assert uniform_timeseries_length(data)

    def test_uniform_length_is_true_if_all_empty_values_at_last_timestep(self):
        data = np.ones((4, 10, 2))
        data[:, -1, :] = 0.0
        assert uniform_timeseries_length(data)

    def test_uniform_length_if_false_if_different_empty_value_locations(self):
        data = np.ones((4, 10, 2))
        data[0, -1, :] = 0.0
        data[1, -2, :] = 0.0
        assert not uniform_timeseries_length(data)

    def test_empty_values_must_be_contiguous_for_uniform_length(self):
        data = np.ones((4, 10, 2))
        data[0, :-1, :] = 0.0
        assert uniform_timeseries_length(data)
