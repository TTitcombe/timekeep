"""
Checks to perform on data
"""
import numpy as np


def is_timeseries(data):
    """
    Check if data is in a timeseries format.
    Timeseries data has 3 dimensions:
      N - number of data points
      T - number of time points
      D - number of data dimensions
    """

    assert len(data.shape) == 3
    return data


def is_shape(data, shape):
    """
    Compare the shape of a dataset
    """
    assert len(data.shape) == len(shape)

    shape_comparison = [
        True if dim2 == -1 else dim1 == dim2 for dim1, dim2 in zip(data.shape, shape)
    ]
    assert all(shape_comparison)

    return data


def none_missing(data):
    """
    Check that no NaN values are present in the data
    """
    assert not np.isnan(data).any()
    return data


def full_timeseries(data, empty_value=0.0):
    """Checks that each timeseries in a dataset is "full"
    i.e. it continues to the last value.

    Empty data values are assumed to be 0.0."""
    if np.isnan(empty_value):
        assert not np.isnan(data).any()
    else:
        assert not (data[:, -1, :] == empty_value).any()
    return data


def at_least_n_datapoints(data, n):
    """
    Check that data has n or more datapoints (first dimension).
    NOTE: This check is in the range [n,). It includes n
    """
    assert data.shape[0] >= n
    return data


def fewer_than_n_datapoints(data, n):
    """
    Check that data has fewer than n datapoints (first dimension).
    NOTE: This check is in the range [1, n). It does not include n
    """
    assert data.shape[0] < n
    return data
