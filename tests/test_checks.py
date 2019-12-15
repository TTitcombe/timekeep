"""
Unit tests for timekeep.checks
"""
import numpy as np
import pandas as pd
import pytest

from timekeep.checks import *
from timekeep.exceptions import TimekeepCheckError


class TestChecks:
    def test_is_timeseries_does_not_raise_1d_timeseries_data(self):
        data = np.random.random((10, 5, 1))
        is_timeseries_dataset(data)

    def test_is_timeseries_does_not_raise_for_timeseries_data(self):
        data = np.random.random((10, 5, 3))
        is_timeseries_dataset(data)

    def test_is_timeseries_does_not_raise_for_timeseries_dataset_of_zeroes(self):
        data = np.random.random((0, 0, 0))
        is_timeseries_dataset(data)

    def test_is_timeseries_does_not_raise_for_empty_timeseries_dataset(self):
        data = np.empty((10, 9, 8))
        is_timeseries_dataset(data)

    def test_is_timeseries_raises_for_sklearn_data(self):
        data = np.random.random((10, 15))
        with pytest.raises(TimekeepCheckError):
            is_timeseries_dataset(data)

    def test_is_timeseries_raises_for_pandas_dataframe(self):
        df = pd.DataFrame({"A": np.random.random((10,)), "B": np.random.random((10,))})
        with pytest.raises(TimekeepCheckError):
            is_timeseries_dataset(df)

    def test_is_flat_dataset_raises_if_data_is_not_pandas_dataframe(self):
        data = np.random.random((102, 4))
        with pytest.raises(TimekeepCheckError):
            is_flat_dataset(data)

    def test_is_flat_dataset_raises_if_data_has_fewer_than_three_columns(self):
        data = pd.DataFrame({"id": [0, 1, 0, 1], "time": [0, 0, 1, 1]})
        with pytest.raises(TimekeepCheckError):
            is_flat_dataset(data)

    def test_is_flat_dataset_raises_if_data_does_not_have_id_column(self):
        data = pd.DataFrame(
            {"not_id": [0, 1, 0, 1], "time": [0, 0, 1, 1], "value": [1, 2, 3, 4]}
        )
        with pytest.raises(TimekeepCheckError):
            is_flat_dataset(data)

    def test_is_flat_dataset_raises_if_data_does_not_have_time_column(self):
        data = pd.DataFrame(
            {"id": [0, 1, 0, 1], "not_time": [0, 0, 1, 1], "value": [1, 2, 3, 4]}
        )
        with pytest.raises(TimekeepCheckError):
            is_flat_dataset(data)

    def test_is_flat_dataset_accepts_flat_dataset(self):
        data = pd.DataFrame(
            {"id": [0, 1, 0, 1], "time": [0, 0, 1, 1], "value": [1, 2, 3, 4]}
        )
        is_flat_dataset(data)

    def test_is_stacked_dataset_raises_if_data_is_not_pandas_dataframe(self):
        data = np.random.random((102, 4))
        with pytest.raises(TimekeepCheckError):
            is_stacked_dataset(data)

    def test_is_stacked_dataset_raises_if_data_does_not_have_four_columns(self):
        data = pd.DataFrame(
            {"id": [0, 1, 0, 1], "time": [0, 0, 1, 1], "value": [1, 2, 3, 4]}
        )
        with pytest.raises(TimekeepCheckError):
            is_stacked_dataset(data)

    def test_is_stacked_dataset_raises_if_data_does_not_have_id_column(self):
        data = pd.DataFrame(
            {
                "not_id": [0, 1, 0, 1],
                "time": [0, 0, 1, 1],
                "kind": [1, 1, 1, 1],
                "value": [1, 2, 3, 4],
            }
        )
        with pytest.raises(TimekeepCheckError):
            is_stacked_dataset(data)

    def test_is_stacked_dataset_raises_if_data_does_not_have_time_column(self):
        data = pd.DataFrame(
            {
                "id": [0, 1, 0, 1],
                "not_time": [0, 0, 1, 1],
                "kind": [1, 1, 1, 1],
                "value": [1, 2, 3, 4],
            }
        )
        with pytest.raises(TimekeepCheckError):
            is_stacked_dataset(data)

    def test_is_stacked_dataset_raises_if_data_does_not_have_kind_column(self):
        data = pd.DataFrame(
            {
                "id": [0, 1, 0, 1],
                "time": [0, 0, 1, 1],
                "not_kind": [1, 1, 1, 1],
                "value": [1, 2, 3, 4],
            }
        )
        with pytest.raises(TimekeepCheckError):
            is_stacked_dataset(data)

    def test_is_stacked_dataset_raises_if_data_does_not_have_value_column(self):
        data = pd.DataFrame(
            {
                "id": [0, 1, 0, 1],
                "time": [0, 0, 1, 1],
                "kind": [1, 1, 1, 1],
                "not_value": [1, 2, 3, 4],
            }
        )
        with pytest.raises(TimekeepCheckError):
            is_stacked_dataset(data)

    def test_is_stacked_dataset_accepts_stacked_dataset(self):
        data = pd.DataFrame(
            {
                "id": [0, 1, 0, 1],
                "time": [0, 0, 1, 1],
                "kind": [1, 1, 1, 1],
                "value": [1, 2, 3, 4],
            }
        )
        is_stacked_dataset(data)

    def test_is_sklearn_dataset_raises_if_not_dataframe_or_array(self):
        # Passes
        is_sklearn_dataset(np.random.random((10, 2)))
        is_sklearn_dataset(pd.DataFrame(np.random.random((10, 2))))

        with pytest.raises(TimekeepCheckError):
            is_sklearn_dataset([[0, 1], [1, 2], [2, 3]])

    def test_is_sklearn_dataset_raises_if_more_than_two_dimensions(self):
        with pytest.raises(TimekeepCheckError):
            is_sklearn_dataset(np.random.random((10, 3, 1)))

    def test_is_sklearn_dataset_raises_if_is_tsfresh_flat_dataset(self):
        data = pd.DataFrame({"id": [0, 0], "time": [0, 1], "value": [100, 102]})

        with pytest.raises(TimekeepCheckError):
            is_sklearn_dataset(data)

    def test_is_sklearn_dataset_raises_if_is_tsfresh_stacked_dataset(self):
        data = pd.DataFrame(
            {
                "id": [0, 0, 0, 0],
                "time": [0, 1, 0, 1],
                "kind": [0, 0, 1, 1],
                "value": [1, 2, 3, 4],
            }
        )

        with pytest.raises(TimekeepCheckError):
            is_sklearn_dataset(data)

    def test_is_shape_returns_true_for_equal_shapes(self):
        data = np.random.random((10, 5, 2))
        is_shape(data, (10, 5, 2))

    def test_is_shape_raises_for_non_equal_dims(self):
        data = np.random.random((10, 5))
        with pytest.raises(TimekeepCheckError):
            is_shape(data, (10, 5, 2))

    def test_is_shape_raises_for_non_equal_shapes(self):
        data = np.random.random((10, 5, 2))
        with pytest.raises(TimekeepCheckError):
            is_shape(data, (10, 5, 3))

    def test_is_shape_can_accept_placeholder_dims(self):
        data = np.random.random((10, 8, 7))
        is_shape(data, (-1, 8, 7))

    def test_none_missing_raises_if_has_a_single_nan(self):
        data = np.random.random((5, 4, 3))
        data[0, -1, 0] = np.nan
        with pytest.raises(TimekeepCheckError):
            none_missing(data)

    def test_none_missing_raises_does_not_raise_if_no_nans(self):
        data = np.random.random((5, 4, 3))
        none_missing(data)

    def test_full_timeseries_raises_if_last_element_is_nan(self):
        data = np.random.random((5, 5, 3))
        data[-1, -1, 1] = np.nan
        with pytest.raises(TimekeepCheckError):
            full_timeseries(data)

    def test_full_timeseries_can_accept_zero_as_empty_value(self):
        data = np.random.random((5, 5, 3))
        data[-1, -1, 1] = 0.0
        with pytest.raises(TimekeepCheckError):
            full_timeseries(data, empty_value=0.0)

    def test_full_timeseries_can_accept_different_empty_value(self):
        data = np.random.random((5, 5, 3))
        data[-1, -1, 0] = 1e8
        with pytest.raises(TimekeepCheckError):
            full_timeseries(data, empty_value=1e8)

    def test_full_timeseries_does_not_raise_if_no_empty_values_at_end(self):
        data = np.random.random((5, 5, 2))
        data[:, -1, :] = 0.0  # should not raise as empty_value is NaN
        full_timeseries(data)

    def test_full_timeseries_does_not_raise_if_empty_values_occur_before_end(self):
        data = np.random.random((5, 5, 2))
        data[:, -2, :] = np.nan
        full_timeseries(data)

    def test_at_least_n_raises_if_fewer_than_n_datapoint(self):
        data = np.random.random((5, 100, 1))

        with pytest.raises(TimekeepCheckError):
            at_least_n_datapoints(data, 6)

    def test_at_least_n_does_not_raise_if_more_than_n_datapoints(self):
        data = np.random.random((5, 100, 1))

        at_least_n_datapoints(data, 4)

    def test_at_least_n_does_not_raise_if_n_datapoints(self):
        data = np.random.random((5, 100, 1))

        at_least_n_datapoints(data, 5)

    def test_fewer_than_n_raises_if_more_than_n_datapoints(self):
        data = np.random.random((5, 100, 1))

        with pytest.raises(TimekeepCheckError):
            fewer_than_n_datapoints(data, 4)

    def test_fewer_than_n_does_not_raise_if_fewer_than_n_datapoints(self):
        data = np.random.random((5, 100, 1))

        fewer_than_n_datapoints(data, 6)

    def test_fewer_than_n_raises_if_n_datapoints(self):
        data = np.random.random((5, 100, 1))

        with pytest.raises(TimekeepCheckError):
            fewer_than_n_datapoints(data, 5)

    def test_has_uniform_length_is_true_if_no_empty_values(self):
        data = np.ones((4, 10, 2))
        has_uniform_length(data)

    def test_has_uniform_length_is_true_if_all_empty_values_at_last_timestep(self):
        data = np.ones((4, 10, 2))
        data[:, -1, :] = np.nan
        has_uniform_length(data)

    def test_has_uniform_length_raises_if_different_empty_value_locations(self):
        data = np.ones((4, 10, 2))
        data[0, -1, :] = np.nan

        with pytest.raises(TimekeepCheckError):
            has_uniform_length(data)

    def test_empty_values_must_be_contiguous_for_uniform_length(self):
        data = np.ones((4, 10, 2))
        data[0, :-1, :] = np.nan
        has_uniform_length(data)
