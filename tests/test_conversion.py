"""
Unit tests for timekeep.conversion
"""
import numpy as np
import pandas as pd
import pytest
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
        self
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

        assert_frame_equal(converted_data, expected_data.astype("float64"))

    def test_to_flat_dataset_converts_tslearn_dataset(self):
        data = np.random.random((2, 3, 2))
        converted_data = to_flat_dataset(data)

        assert converted_data.shape == (6, 4)
        assert list(converted_data.columns) == ["id", "time", "0", "1"]

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
