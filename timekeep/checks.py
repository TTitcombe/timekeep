"""
Checks to perform on data
"""
import numpy as np
import pandas as pd

from .utility import find_stop_indices


def is_timeseries_dataset(data):
    """
    Check if data is in tslearn-style timeseries format.

    Timeseries data has three dimensions:
        N - the number of data points
        T - the number of points in time
        D - the number of parameters

    Parameters
    ----------
    data : np.ndarray
        The data to check

    Raises
    ------
    AssertionError
        If the data shape does not have length 3
    """

    assert len(data.shape) == 3


def is_flat_dataset(data):
    """
    Check if data is tsfresh-style flat dataframe format.

    Flat dataframes are pandas DataFrame object with the
    following columns:
        id - The id of the timeseries to which the datapoint relates
        time - Time value of the datapoint
    There can be any number of "value" columns relating to different parameters
    of the timeseries

    Parameters
    ----------
    data
        The data to check

    Raises
    ------
    AssertionError
        data is not a pandas DataFrame
        data does not have more than 2 columns
        data does not have an "id" column
        data does not have a "time" column

    Notes
    -----
    https://tsfresh.readthedocs.io/en/latest/text/data_formats.html
    """
    assert isinstance(data, pd.DataFrame)
    assert data.shape[1] > 2
    assert "id" in data.columns
    assert "time" in data.columns


def is_stacked_dataset(data):
    """
    Check if data is tsfresh-style stacked dataframe format.

    Stacked dataframes are pandas DataFrame object with the
    following columns:
        id - The id of the timeseries to which the datapoint relates
        time - Time value of the datapoint
        kind - The value to which the datapoint relates
        value - The value of the datapoint

    Parameters
    ----------
    data
        The data to check

    Raises
    ------
    AssertionError
        data is not a pandas DataFrame
        data does not have 4 columns
        data does not have an "id" column
        data does not have a "time" column
        data does not have a "kind" column
        data doe not have a "value" column

    Notes
    -----
    https://tsfresh.readthedocs.io/en/latest/text/data_formats.html
    """
    assert isinstance(data, pd.DataFrame)
    assert data.shape[1] == 4
    assert "id" in data.columns
    assert "time" in data.columns
    assert "kind" in data.columns
    assert "value" in data.columns


def is_shape(data, shape):
    """
    Compare the shape of a dataset to a given shape.

    -1 can be used in a shape dimension to denote that that
    dimension can have any number of data points.

    Parameters
    ----------
    data : np.ndarray
        The data to check
    shape : tuple
        The shape to compare data to

    Raises
    ------
    AssertionError
        If the data shape does not match provided shape
    """
    assert len(data.shape) == len(shape)

    shape_comparison = [
        True if dim2 == -1 else dim1 == dim2 for dim1, dim2 in zip(data.shape, shape)
    ]
    assert all(shape_comparison)


def none_missing(data):
    """
    Check that no NaN values are present in the data

    Parameters
    ----------
    data : np.ndarray
        The data to check

    Raises
    ------
    AssertionError
        If data contain any NaN values
    """
    assert not np.isnan(data).any()


def full_timeseries(data, empty_value=np.nan):
    """
    Check that each timeseries in a dataset is "full"
    i.e. it continues to the last time step.

    Empty values, to denote that a timeseries has finished,
    are assumped to be np.nan values.

    Parameters
    ----------
    data : np.ndarray
        The data to check
    empty_value
        The value which denotes empty data / the timeseries has ended

    Raises
    ------
    AssertionError
        If empty values are present at the end of one or
        more timeseries in the data"""
    last_data = data[:, -1, :]
    if np.isnan(empty_value):
        assert not np.isnan(last_data).any()
    else:
        assert not (last_data == empty_value).any()


def at_least_n_datapoints(data, n):
    """
    Check that data has n or more datapoints (first dimension)

    Parameters
    ----------
    data : np.ndarray
        The data to check
    n : int
        The minimum number of datapoints required

    Raises
    ------
    AssertionError
        If data has fewer than n datapoints
    """
    assert data.shape[0] >= n


def fewer_than_n_datapoints(data, n):
    """
    Check that data has fewer than n datapoints (first dimension)

    Parameters
    ----------
    data : np.ndarray
        The data to check
    n : int
        The number of datapoints data must have fewer than

    Raises
    ------
    AssertionError
        If data has n or more datapoints
    """
    assert data.shape[0] < n


def check_uniform_length(data):
    lengths = find_stop_indices(data, empty_value=np.nan)
    """
    Check that each timeseries in a dataset has the same length

    Parameters
    ----------
    data : np.ndarray
        The data to check

    Raises
    ------
    AssertionError
        If data contains timeseries with different lengths
    """
    assert np.unique(lengths).size == 1
