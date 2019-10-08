"""
Unit tests for timekeep.decorators
"""
import numpy as np
import pandas as pd
import pytest

from timekeep.decorators import *


class TestDecorators:
    def test_is_timeseries_returns_data_if_timeseries_shape(self):
        @is_timeseries
        def inner_func():
            return np.random.random((2, 10, 1))

        data = inner_func()

        assert data.shape == (2, 10, 1)

    def test_is_timeseries_raises_if_data_not_timeseries_shape(self):
        @is_timeseries
        def inner_func():
            return np.random.random((2, 10))

        with pytest.raises(AssertionError):
            data = inner_func()

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
