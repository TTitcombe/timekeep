"""
Unit tests for timekeep.checks
"""
import numpy as np
import pandas as pd
import pytest

from timekeep.checks import *


class TestChecks:
    def test_is_timeseries_does_not_raise_1d_timeseries_data(self):
        data = np.random.random((10, 5, 1))
        is_timeseries(data)

    def test_is_timeseries_does_not_raise_for_timeseries_data(self):
        data = np.random.random((10, 5, 3))
        is_timeseries(data)

    def test_is_timeseries_does_not_raise_for_timeseries_dataset_of_zeroes(self):
        data = np.random.random((0, 0, 0))
        is_timeseries(data)

    def test_is_timeseries_does_not_raise_for_empty_timeseries_dataset(self):
        data = np.empty((10, 9, 8))
        is_timeseries(data)

    def test_is_timeseries_raises_for_sklearn_data(self):
        data = np.random.random((10, 15))
        with pytest.raises(AssertionError):
            is_timeseries(data)

    def test_is_timeseries_raises_for_pandas_dataframe(self):
        df = pd.DataFrame({"A": np.random.random((10,)), "B": np.random.random((10,))})
        with pytest.raises(AssertionError):
            is_timeseries(df)

    def test_is_shape_returns_true_for_equal_shapes(self):
        data = np.random.random((10, 5, 2))
        is_shape(data, (10, 5, 2))

    def test_is_shape_raises_for_non_equal_dims(self):
        data = np.random.random((10, 5))
        with pytest.raises(AssertionError):
            is_shape(data, (10, 5, 2))

    def test_is_shape_raises_for_non_equal_shapes(self):
        data = np.random.random((10, 5, 2))
        with pytest.raises(AssertionError):
            is_shape(data, (10, 5, 3))

    def test_is_shape_can_accept_placeholder_dims(self):
        data = np.random.random((10, 8, 7))
        is_shape(data, (-1, 8, 7))

    def test_none_missing_raises_if_has_a_single_nan(self):
        data = np.random.random((5, 4, 3))
        data[0, -1, 0] = np.nan
        with pytest.raises(AssertionError):
            none_missing(data)

    def test_none_missing_raises_does_not_raise_if_no_nans(self):
        data = np.random.random((5, 4, 3))
        none_missing(data)

    def test_full_timeseries_raises_if_last_element_is_zero(self):
        data = np.random.random((5, 5, 3))
        data[-1, -1, 1] = 0.0
        with pytest.raises(AssertionError):
            full_timeseries(data)

    def test_full_timeseries_can_accept_nan_as_empty_value(self):
        data = np.random.random((5, 5, 3))
        data[-1, -1, 1] = np.nan
        with pytest.raises(AssertionError):
            full_timeseries(data, empty_value=np.nan)

    def test_full_timeseries_can_accept_different_empty_value(self):
        data = np.random.random((5, 5, 3))
        data[-1, -1, 0] = 1e8
        with pytest.raises(AssertionError):
            full_timeseries(data, empty_value=1e8)

    def test_full_timeseries_does_not_raise_if_no_empty_values_at_end(self):
        data = np.random.random((5, 5, 2))
        data[:, -2, :] = 0.0  # should not raise
        full_timeseries(data)

    def test_at_least_n_raises_if_fewer_than_n_datapoint(self):
        data = np.random.random((5, 100, 1))

        with pytest.raises(AssertionError):
            at_least_n_datapoints(data, 6)

    def test_at_least_n_does_not_raise_if_more_than_n_datapoints(self):
        data = np.random.random((5, 100, 1))

        at_least_n_datapoints(data, 4)

    def test_at_least_n_does_not_raise_if_n_datapoints(self):
        data = np.random.random((5, 100, 1))

        at_least_n_datapoints(data, 5)

    def test_fewer_than_n_raises_if_more_than_n_datapoints(self):
        data = np.random.random((5, 100, 1))

        with pytest.raises(AssertionError):
            fewer_than_n_datapoints(data, 4)

    def test_fewer_than_n_does_not_raise_if_fewer_than_n_datapoints(self):
        data = np.random.random((5, 100, 1))

        fewer_than_n_datapoints(data, 6)

    def test_fewer_than_n_raises_if_n_datapoints(self):
        data = np.random.random((5, 100, 1))

        with pytest.raises(AssertionError):
            fewer_than_n_datapoints(data, 5)
