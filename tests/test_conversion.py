"""
Unit tests for timekeep.conversion
"""
import numpy as np
import pytest
from sklearn.decomposition import PCA

from timekeep.conversion import *


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
