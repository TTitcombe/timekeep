"""
Checks to perform on data
"""
import numpy as np
import pandas as pd

from .exceptions import TimekeepCheckError
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
    TimekeepCheckError
        If the data shape does not have length 3
    """

    if not len(data.shape) == 3:
        raise TimekeepCheckError(
            "is_timeseries_dataset: data of shape {} has {} axes, not 3".format(
                data.shape, len(data.shape)
            )
        )


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
    TimekeepCheckError
        data is not a pandas DataFrame
        data does not have more than 2 columns
        data does not have an "id" column
        data does not have a "time" column

    Notes
    -----
    https://tsfresh.readthedocs.io/en/latest/text/data_formats.html
    """
    if not isinstance(data, pd.DataFrame):
        raise TimekeepCheckError(
            "is_flat_dataset: data of type {} is not pandas.DataFrame".format(
                type(data)
            )
        )

    if not data.shape[1] > 2:
        raise TimekeepCheckError(
            "is_flat_dataset: data must have at least "
            "2 columns; data has {}".format(data.shape[1])
        )

    if "kind" in data.columns:
        raise TimekeepCheckError(
            "is_flat_dataset: data contains a 'kind' column."
            "It is probably a stacked dataset"
        )

    if "id" not in data.columns:
        raise TimekeepCheckError("is_flat_dataset: data does not contain 'id' column")

    if "time" not in data.columns:
        raise TimekeepCheckError("is_flat_dataset: data does not contain 'time' column")


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
    TimekeepCheckError
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
    if not isinstance(data, pd.DataFrame):
        raise TimekeepCheckError(
            "is_stacked_dataset: data of type {} is not pandas.Dataframe".format(
                type(data)
            )
        )

    if not data.shape[1] == 4:
        raise TimekeepCheckError(
            "is_stacked_dataset: data must have 4 columns; data has {}".format(
                data.shape[1]
            )
        )

    if "id" not in data.columns:
        raise TimekeepCheckError(
            "is_stacked_dataset: data does not contain 'id' column"
        )

    if "time" not in data.columns:
        raise TimekeepCheckError(
            "is_stacked_dataset: data does not contain 'time' column"
        )

    if "kind" not in data.columns:
        raise TimekeepCheckError(
            "is_stacked_dataset: data does not contain 'kind' column"
        )

    if "value" not in data.columns:
        raise TimekeepCheckError(
            "is_stacked_dataset: data does not contain 'value' column"
        )


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
    TimekeepCheckError
        If the data shape does not match provided shape
    """
    if not len(data.shape) == len(shape):
        raise TimekeepCheckError(
            "is_shape: data does not have correct number of axes; "
            "data has {} axes".format(len(data.shape))
        )

    shape_comparison = [
        True if dim2 == -1 else dim1 == dim2 for dim1, dim2 in zip(data.shape, shape)
    ]
    if not all(shape_comparison):
        raise TimekeepCheckError(
            "is_shape: data has shape {}; does not match shape {}".format(
                data.shape, shape
            )
        )


def none_missing(data):
    """
    Check that no NaN values are present in the data

    Parameters
    ----------
    data : np.ndarray
        The data to check

    Raises
    ------
    TimekeepCheckError
        If data contain any NaN values
    """
    if np.isnan(data).any():
        raise TimekeepCheckError("none_missing: data contains NaN values")


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
    TimekeepCheckError
        If empty values are present at the end of one or
        more timeseries in the data"""
    last_data = data[:, -1, :]
    if np.isnan(empty_value):
        if np.isnan(last_data).any():
            raise TimekeepCheckError("full_timeseries: timeseries are not full")
    else:
        if (last_data == empty_value).any():
            raise TimekeepCheckError("full_timeseries: timeseries are not full")


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
    TimekeepCheckError
        If data has fewer than n datapoints
    """
    if not data.shape[0] >= n:
        raise TimekeepCheckError(
            "at_least_n_datapoints: data has fewer than n datapoints; "
            "data has {} datapoints".format(data.shape[0])
        )


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
    TimekeepCheckError
        If data has n or more datapoints
    """
    if not data.shape[0] < n:
        raise TimekeepCheckError(
            "fewer_than_n_datapoints: data has n or more datapoints;"
            "data has {} datapoints".format(data.shape[0])
        )


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
    TimekeepCheckError
        If data contains timeseries with different lengths
    """
    if not np.unique(lengths).size == 1:
        raise TimekeepCheckError(
            "check_uniform_length: timeseries in data are not uniform length"
        )
