"""
Unit tests for timekeep.conversion
"""
import numpy as np
import pytest
from sklearn.base import BaseEstimator

from timekeep.conversion import *


@accept_timeseries_input
class DummyEstimator(BaseEstimator):
    def fit(self, x):
        return x

    def transform(self, x):
        return x

    def fit_transform(self, x):
        return x

    def predict(self, x):
        return x


class TestConversion:
    def test_accept_timeseries_input_augments_fit_method(self):
        estimator = DummyEstimator()
        data = estimator.fit(np.random.random((10, 15, 2)))
        assert data.shape == (10, 30)

    def test_accept_timeseries_input_augments_transform_method(self):
        estimator = DummyEstimator()
        data = estimator.transform(np.random.random((10, 15, 2)))
        assert data.shape == (10, 30)

    def test_accept_timeseries_input_augments_fit_transform_method(self):
        estimator = DummyEstimator()
        data = estimator.fit_transform(np.random.random((10, 15, 2)))
        assert data.shape == (10, 30)

    def test_accept_timeseries_input_augments_predict_method(self):
        estimator = DummyEstimator()
        data = estimator.predict(np.random.random((10, 15, 2)))
        assert data.shape == (10, 30)
