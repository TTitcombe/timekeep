"""
Unit tests for timekeep.conversion
"""
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_frame_equal
from sklearn.decomposition import PCA

from timekeep.conversion import *
from timekeep.exceptions import TimekeepCheckError


@timeseries_transformer
class DummyTransformer(PCA):
    pass


class TestTimeseriesTransformer:
    def test_accept_timeseries_input_augments_fit_method(self):
        estimator = DummyTransformer(n_components=2)
        estimator.fit(np.random.random((10, 15, 2)))

    def test_accept_timeseries_input_augments_transform_method(self):
        estimator = DummyTransformer(n_components=2)

        fit_data = np.random.random((10, 15, 5))
        estimator.fit(fit_data)
        data = estimator.transform(fit_data)
        assert data.shape == (10, 2)

    def test_accept_timeseries_input_augments_fit_transform_method(self):
        estimator = DummyTransformer(n_components=2)
        data = estimator.fit_transform(np.random.random((10, 15, 10)))
        assert data.shape == (10, 2)

    def test_convert_timeseries_input_works_on_functions(self):
        def load(data):
            return data

        func = convert_timeseries_input(load)
        returned_data = func(np.random.random((10, 4, 2)))
        assert returned_data.shape == (10, 8)

    def test_convert_output_to_timeseries_returns_timeseries(self):
        @convert_output_to_timeseries
        def load():
            return np.random.random((10, 5, 2))

        # This should return a timeseries dataset as normal
        returned_data = load()
        assert returned_data.shape == (10, 5, 2)

    def test_convert_output_to_timeseries_converts_two_dimensional_output(self):
        @convert_output_to_timeseries
        def load():
            return np.random.random((10, 5))

        returned_data = load()
        assert returned_data.shape == (10, 5, 1)

    def test_convert_output_to_timeseries_raises_if_output_is_more_than_three_dimensional(
        self,
    ):
        @convert_output_to_timeseries
        def load():
            return np.random.random((5, 4, 3, 2))

        with pytest.raises(TimekeepCheckError):
            load()

    def test_to_flat_dataset_can_accept_flat_dataset(self):
        data = pd.DataFrame(
            {
                "id": [0, 0, 0, 1, 1, 1],
                "time": [0, 1, 2, 0, 1, 2],
                "value_1": [1, 2, 3, 4, 5, 6],
                "value_2": [10, 9, 8, 7, 6, 5],
            }
        )

        converted_data = to_flat_dataset(data)
        assert_frame_equal(data, converted_data)

    def test_to_flat_dataset_converts_stacked_dataset(self):
        data = pd.DataFrame(
            {
                "id": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                "time": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                "kind": [
                    "value_1",
                    "value_1",
                    "value_1",
                    "value_2",
                    "value_2",
                    "value_2",
                    "value_1",
                    "value_1",
                    "value_1",
                    "value_2",
                    "value_2",
                    "value_2",
                ],
                "value": [1, 2, 3, 4, 5, 6, 10, 9, 8, 7, 6, 5],
            }
        )

        expected_data = pd.DataFrame(
            {
                "id": [0, 0, 0, 1, 1, 1],
                "time": [0, 1, 2, 0, 1, 2],
                "value_1": [1, 2, 3, 10, 9, 8],
                "value_2": [4, 5, 6, 7, 6, 5],
            }
        )
        converted_data = to_flat_dataset(data)

        assert_frame_equal(converted_data, expected_data)

    def test_to_flat_dataset_converts_tslearn_dataset(self):
        data = np.random.random((2, 3, 2))
        converted_data = to_flat_dataset(data)

        assert converted_data.shape == (6, 4)
        assert list(converted_data.columns) == ["id", "time", "0", "1"]

    def test_to_flat_dataset_raises_value_error_if_data_format_not_recognised(self):
        with pytest.raises(ValueError):
            data = to_flat_dataset(np.random.random((10,)))

    def test_flat_to_stacked_to_flat_returns_same_dataframe(self):
        data = pd.DataFrame(
            {
                "id": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                "time": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                "kind": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            }
        )
        converted_data = to_stacked_dataset(to_flat_dataset(data))
        assert_frame_equal(
            converted_data.sort_values(["id", "kind", "time"]).reset_index(drop=True),
            data,
        )

    def test_to_stacked_dataset_can_accept_stacked_dataset(self):
        data = pd.DataFrame(
            {
                "id": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                "time": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                "kind": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            }
        )

        converted_data = to_stacked_dataset(data)

        assert_frame_equal(converted_data, data)

    def test_to_stacked_dataset_converts_flat_dataset(self):
        data = pd.DataFrame(
            {
                "id": [0, 0, 0, 1, 1, 1],
                "time": [0, 1, 2, 0, 1, 2],
                "value_1": [1, 2, 3, 4, 5, 6],
                "value_2": [10, 9, 8, 7, 6, 5],
            }
        )

        converted_data = to_stacked_dataset(data)
        expected_data = pd.DataFrame(
            {
                "id": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                "time": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                "kind": [
                    "value_1",
                    "value_1",
                    "value_1",
                    "value_1",
                    "value_1",
                    "value_1",
                    "value_2",
                    "value_2",
                    "value_2",
                    "value_2",
                    "value_2",
                    "value_2",
                ],
                "value": [1, 2, 3, 4, 5, 6, 10, 9, 8, 7, 6, 5],
            }
        )

        assert_frame_equal(converted_data, expected_data)

    def test_to_stacked_dataset_converts_tslearn_dataset(self):
        data = np.random.random((10, 25, 3))
        converted_data = to_stacked_dataset(data)

        assert converted_data.shape == (750, 4)
        assert list(converted_data.columns) == ["id", "time", "kind", "value"]

    def test_to_stacked_dataset_raises_value_error_if_data_format_not_recognised(self):
        with pytest.raises(ValueError):
            data = to_stacked_dataset(np.random.random((10,)))

    def test_stacked_to_flat_to_stacked_returns_same_dataframe(self):
        data = pd.DataFrame(
            {
                "id": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
                "time": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                "kind": [
                    "value_1",
                    "value_1",
                    "value_1",
                    "value_1",
                    "value_1",
                    "value_1",
                    "value_2",
                    "value_2",
                    "value_2",
                    "value_2",
                    "value_2",
                    "value_2",
                ],
                "value": [1, 2, 3, 4, 5, 6, 10, 9, 8, 7, 6, 5],
            }
        )
        converted_data = to_stacked_dataset(to_flat_dataset(data))
        assert_frame_equal(converted_data, data)

    def test_to_timeseries_dataset_converts_flat_dataset(self):
        data = pd.DataFrame(
            {
                "id": [0, 1, 0, 1],
                "time": [0, 0, 1, 1],
                "value1": [1, 2, 3, 4],
                "value2": [5, 6, 7, 8],
            }
        )

        converted_data = to_timeseries_dataset(data)

        assert isinstance(converted_data, np.ndarray)
        assert_array_equal(
            converted_data, np.array([[[1, 5], [3, 7]], [[2, 6], [4, 8]]])
        )

    def test_to_timeseries_dataset_converts_stacked_dataset(self):
        data = pd.DataFrame(
            {
                "id": [0, 1, 0, 1, 0, 1, 0, 1],
                "time": [0, 0, 1, 1, 0, 0, 1, 1],
                "kind": [0, 0, 0, 0, 1, 1, 1, 1],
                "value": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )

        converted_data = to_timeseries_dataset(data)

        assert isinstance(converted_data, np.ndarray)
        assert_array_equal(
            converted_data, np.array([[[1, 5], [3, 7]], [[2, 6], [4, 8]]])
        )

    def test_to_timeseries_dataset_converts_sklearn_dataset_with_all_dims_provided(
        self,
    ):
        data = pd.DataFrame(np.arange(120).reshape((10, 12)))
        converted_data = to_timeseries_dataset(data, t=6, d=2)

        assert converted_data.shape == (10, 6, 2)
        assert_array_equal(converted_data, np.arange(120).reshape((10, 6, 2)))

    def test_to_timeseries_dataset_converts_sklearn_dataset_with_t_provided(self):
        data = pd.DataFrame(np.arange(120).reshape((10, 12)))
        converted_data = to_timeseries_dataset(data, t=4)

        assert converted_data.shape == (10, 4, 3)
        assert_array_equal(converted_data, np.arange(120).reshape((10, 4, 3)))

    def test_to_timeseries_dataset_converts_sklearn_dataset_with_d_provided(self):
        data = pd.DataFrame(np.arange(120).reshape(10, 12))
        converted_data = to_timeseries_dataset(data, d=3)

        assert converted_data.shape == (10, 4, 3)
        assert_array_equal(converted_data, np.arange(120).reshape((10, 4, 3)))

    def test_to_timeseries_dataset_converts_sklearn_dataset_with_d_equal_to_one_when_no_dims_provided(
        self,
    ):
        data = np.arange(120).reshape(10, 12)
        converted_data = to_timeseries_dataset(data)

        assert converted_data.shape == (10, 12, 1)
        assert_array_equal(
            converted_data, np.expand_dims(np.arange(120).reshape((10, 12)), axis=2)
        )

    def test_to_timeseries_dataset_raises_value_error_if_data_format_not_recognised(
        self,
    ):
        with pytest.raises(ValueError):
            data = to_timeseries_dataset(np.random.random((10,)))

    def test_timeseries_to_sklearn_to_timeseries_return_same_array(self):
        data = np.random.random((18, 26, 4))
        converted_data = to_timeseries_dataset(to_sklearn_dataset(data), d=4)
        assert_array_equal(converted_data, data)

    def test_to_sklearn_dataset_converts_timeseries_dataset(self):
        datum = np.expand_dims(np.array([[1, 4], [2, 5], [3, 6]]), axis=0)
        data = np.vstack((datum, datum))

        converted_data = to_sklearn_dataset(data)

        expected_datum = np.expand_dims(np.array([1, 2, 3, 4, 5, 6]), axis=0)
        expected_data = np.vstack((expected_datum, expected_datum))

        assert_array_equal(converted_data, expected_data)

    def test_to_sklearn_dataset_converts_stacked_dataset(self):
        data = pd.DataFrame(
            {
                "id": [0, 1, 0, 1, 0, 1, 0, 1],
                "time": [0, 0, 1, 1, 0, 0, 1, 1],
                "kind": [0, 0, 0, 0, 1, 1, 1, 1],
                "value": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )

        converted_data = to_sklearn_dataset(data)

        expected_data = np.array([[1, 3, 5, 7], [2, 4, 6, 8]])
        assert_array_equal(converted_data, expected_data)

    def test_to_sklearn_dataset_converts_flat_dataset(self):
        data = pd.DataFrame(
            {
                "id": [0, 1, 0, 1],
                "time": [0, 0, 1, 1],
                "value1": [1, 2, 3, 4],
                "value2": [5, 6, 7, 8],
            }
        )

        converted_data = to_sklearn_dataset(data)

        expected_data = np.array([[1, 3, 5, 7], [2, 4, 6, 8]])

        assert_array_equal(converted_data, expected_data)

    def test_to_sklearn_dataset_raises_value_error_if_data_format_not_recognised(self):
        with pytest.raises(ValueError):
            data = to_sklearn_dataset(np.random.random((10,)))

    def test_to_sklearn_dataset_returns_sklearn_dataset(self):
        data = pd.DataFrame(np.random.random((12, 80)))
        converted_data = to_sklearn_dataset(data)

        assert_array_equal(converted_data, data)

    def test_sklearn_to_timeseries_to_sklearn_returns_same_array(self):
        data = np.random.random((18, 104))
        converted_data = to_sklearn_dataset(to_timeseries_dataset(data))
        assert_array_equal(converted_data, data)

    def test_sklearn_to_timeseries_to_sklearn_return_same_array_when_one_dimension_provided(
        self,
    ):
        data = np.random.random((18, 104))
        converted_data = to_sklearn_dataset(to_timeseries_dataset(data, t=26))
        assert_array_equal(converted_data, data)

    def test_sklearn_to_timeseries_to_sklearn_return_same_array_when_all_dimensions_provided(
        self,
    ):
        data = np.random.random((18, 104))
        converted_data = to_sklearn_dataset(to_timeseries_dataset(data, t=26, d=4))
        assert_array_equal(converted_data, data)
