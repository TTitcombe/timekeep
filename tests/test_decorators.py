"""
Unit tests for timekeep.decorators
"""
import numpy as np
import pandas as pd
import pytest

from timekeep.decorators import *


class TestDecorators:
    def test_is_timeseries_returns_data_if_timeseries_shape(self):
        @is_timeseries_dataset
        def inner_func():
            return np.random.random((2, 10, 1))

        data = inner_func()

        assert data.shape == (2, 10, 1)

    def test_is_timeseries_raises_if_data_not_timeseries_shape(self):
        @is_timeseries_dataset
        def inner_func():
            return np.random.random((2, 10))

        with pytest.raises(AssertionError):
            data = inner_func()

    def test_is_flat_dataset_raises_if_data_is_not_pandas_dataframe(self):
        @is_flat_dataset
        def inner_func():
            return np.random.random((15, 3))

        with pytest.raises(AssertionError):
            data = inner_func()

    def test_is_flat_dataset_raises_if_data_has_fewer_than_three_columns(self):
        @is_flat_dataset
        def inner_func():
            return pd.DataFrame({"id": [0, 1], "time": [0, 0]})

        with pytest.raises(AssertionError):
            data = inner_func()

    def test_is_flat_dataset_returns_data_if_flat_dataset(self):
        @is_flat_dataset
        def inner_func():
            return pd.DataFrame({"id": [0, 1], "time": [0, 0], "value": [5, 4]})

        data = inner_func()
        assert data.shape == (2, 3)
        assert list(data.columns) == ["id", "time", "value"]

    def test_is_stacked_dataset_raises_if_data_is_not_pandas_dataframe(self):
        @is_stacked_dataset
        def inner_func():
            return np.random.random((15, 4))

        with pytest.raises(AssertionError):
            data = inner_func()

    def test_is_stacked_dataset_raises_if_data_does_not_have_four_columns(self):
        @is_stacked_dataset
        def inner_func():
            return pd.DataFrame(
                {
                    "id": [0, 1],
                    "time": [0, 0],
                    "kind": [1, 1],
                    "value": [5, 4],
                    "another_col": [1, 1],
                }
            )

        with pytest.raises(AssertionError):
            data = inner_func()

    def test_is_stacked_dataset_returns_data_if_stacked_dataset(self):
        @is_stacked_dataset
        def inner_func():
            return pd.DataFrame(
                {"id": [0, 1], "time": [0, 0], "kind": [1, 1], "value": [5, 4]}
            )

        data = inner_func()
        assert data.shape == (2, 4)
        assert list(data.columns) == ["id", "time", "kind", "value"]

    def test_is_shape_returns_data_if_shapes_match(self):
        @is_shape((1, 2, 3))
        def inner_func():
            return np.random.random((1, 2, 3))

        data = inner_func()
        assert data.shape == (1, 2, 3)

    def test_is_shape_accept_minus_one_as_placeholder(self):
        @is_shape((-1, 2, 3))
        def inner_func(n):
            return np.random.random((n, 2, 3))

        data = inner_func(1)
        assert data.shape == (1, 2, 3)

        data = inner_func(100)
        assert data.shape == (100, 2, 3)

    def test_is_shape_raises_if_shapes_do_not_match(self):
        @is_shape((10, 100, 1))
        def inner_func():
            return np.random.random((10, 55, 1))

        with pytest.raises(AssertionError):
            data = inner_func()

    def test_none_missing_does_not_raise_if_no_nans(self):
        @none_missing
        def inner_func():
            return np.random.random((10, 10, 3))

        data = inner_func()
        assert not np.isnan(data).any()

    def test_none_missing_raises_if_nans_present(self):
        @none_missing
        def inner_func():
            data = np.random.random((10, 10, 3))
            data[-1, -1, -1] = np.nan
            return data

        with pytest.raises(AssertionError):
            data = inner_func()

    def test_full_timeseries_does_not_raise_if_all_timeseries_continue_to_end(self):
        @full_timeseries(0.0)
        def inner_func():
            return np.ones((10, 4, 2))

        data = inner_func()
        assert (data[:, -1, :] != 0.0).all()

    def test_full_timeseries_can_accept_custom_empty_data_value(self):
        empty_value = 1.0

        @full_timeseries(empty_value)
        def inner_func():
            return np.zeros((10, 4, 2))

        data = inner_func()
        assert (data[:, -1, :] != empty_value).all()

    def test_full_timeseries_raises_if_timeseries_different_length(self):
        @full_timeseries(0.0)
        def inner_func():
            data = np.ones((10, 4, 2))
            data[-1, -1, :] = 0.0
            return data

        with pytest.raises(AssertionError):
            data = inner_func()

    def test_at_least_n_raises_if_fewer_than_n(self):
        @at_least_n_datapoints(5)
        def inner_func():
            return np.random.random((4, 10, 1))

        with pytest.raises(AssertionError):
            inner_func()

    def test_at_least_n_does_not_raise_if_n_or_more(self):
        @at_least_n_datapoints(5)
        def inner_func(n):
            return np.random.random((n, 10, 1))

        inner_func(5)
        inner_func(6)

    def test_fewer_than_n_raises_if_n(self):
        @fewer_than_n_datapoints(5)
        def inner_func():
            return np.random.random((5, 100, 2))

        with pytest.raises(AssertionError):
            inner_func()

    def test_fewer_than_n_raises_if_more_than_n(self):
        @fewer_than_n_datapoints(5)
        def inner_func():
            return np.random.random((6, 100, 2))

        with pytest.raises(AssertionError):
            inner_func()

    def test_fewer_than_n_does_not_raise_if_fewer_than_n(self):
        @fewer_than_n_datapoints(5)
        def inner_func():
            return np.random.random((4, 100, 2))

        inner_func()

    def test_datapoints_in_range_raises_if_n_lower_than_range(self):
        @datapoints_in_range(5, 10)
        def inner_func():
            return np.random.random((4, 100, 2))

        with pytest.raises(AssertionError):
            inner_func()

    def test_datapoints_in_range_raises_if_n_greater_than_range(self):
        @datapoints_in_range(5, 10)
        def inner_func():
            return np.random.random((11, 100, 2))

        with pytest.raises(AssertionError):
            inner_func()

    def test_datapoints_in_range_raises_if_n_equal_to_upper_bound(self):
        @datapoints_in_range(5, 10)
        def inner_func():
            return np.random.random((10, 100, 2))

        with pytest.raises(AssertionError):
            inner_func()

    def test_datapoints_in_range_does_not_raise_if_n_equal_to_lower_bound(self):
        @datapoints_in_range(5, 10)
        def inner_func():
            return np.random.random((5, 100, 2))

        inner_func()

    def test_datapoints_in_range_does_not_raise_if_n_in_range(self):
        @datapoints_in_range(5, 10)
        def inner_func():
            return np.random.random((7, 100, 2))

        inner_func()

    def test_uniform_length_does_not_raise_if_no_missing_values(self):
        @uniform_length
        def inner_func():
            return np.random.random((10, 25, 3))

        inner_func()

    def test_uniform_length_raises_if_datapoints_at_different_lengths(self):
        @uniform_length
        def inner_func():
            data = np.random.random((10, 25, 3))
            data[0, -1, :] = np.nan
            data[1, -2:, :] = np.nan
            return data

        with pytest.raises(AssertionError):
            inner_func()
