"""
Unit tests for timekeep.checks
"""
import numpy as np
import pandas as pd
import pytest

from timekeep.checks import *


class TestChecks:
    def test_is_timeseries_returns_true_for_1d_timeseries_data(self):
        data = np.random.random((10, 5, 1))
        assert is_timeseries(data)

    def test_is_timeseries_returns_true_for_timeseries_data(self):
        data = np.random.random((10, 5, 3))
        assert is_timeseries(data)

    def test_is_timeseries_returns_true_for_timeseries_dataset_of_zeroes(self):
        data = np.random.random((0, 0, 0))
        assert is_timeseries(data)

    def test_is_timeseries_returns_true_for_empty_timeseries_dataset(self):
        data = np.empty((10, 9, 8))
        assert is_timeseries(data)

    def test_is_timeseries_returns_false_for_sklearn_data(self):
        data = np.random.random((10, 15))
        assert not is_timeseries(data)

    def test_is_timeseries_returns_false_for_pandas_dataframe(self):
        df = pd.DataFrame({"A": np.random.random((10,)), "B": np.random.random((10,))})
        assert not is_timeseries(df)
