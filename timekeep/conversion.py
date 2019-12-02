"""
Functions for converting between data formats
"""
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from tslearn.utils import to_sklearn_dataset, to_time_series_dataset

from .checks import is_flat_dataset, is_stacked_dataset, is_timeseries_dataset
from .exceptions import TimekeepCheckError


def convert_timeseries_input(func):
    def inner(*args, **kwargs):
        dim = 0
        x = args[dim]  # For functions, x should be first argument
        if not isinstance(x, np.ndarray):
            dim = 1
            x = args[dim]  # For methods, arguments are (self, x, ...)
            assert isinstance(x, np.ndarray)

        x = to_sklearn_dataset(x)
        args = [args[i] if i != dim else x for i in range(len(args))]

        return func(*args, **kwargs)

    return inner


def convert_output_to_timeseries(func):
    def inner(*args, **kwargs):
        data = func(*args, **kwargs)
        if len(data.shape) == 3:
            return data

        # If it's not 2-dimensional, we can't handle it
        if not len(data.shape) == 2:
            raise TimekeepCheckError(
                "convert_output_to_timeseries: data has {} axes; "
                "data must have 2 axes".format(data.shape)
            )
        return to_time_series_dataset(data)

    return inner


def timeseries_transformer(cls: TransformerMixin) -> TransformerMixin:
    """
    Augment sklearn.TransformerMixin classes to accept timeseries datasets

    Parameters
    ----------
    cls : TransformerMixin
        The class to augment

    Returns
    -------
    TransformerMixin
        The input class, which now accepts timeseries datasets as input
    """
    cls.fit = convert_timeseries_input(cls.fit)
    cls.transform = convert_timeseries_input(cls.transform)
    cls.fit_transform = convert_timeseries_input(cls.fit_transform)

    return cls


# ----- Format conversion ----- #
def to_flat_dataset(data):
    """
    Convert a tslearn timeseries or tsfresh stacked dataset
    to a tsfresh flat dataset

    A flat dataset is a DataFrame with columns for 'id' (of timeseries),
    'time' (at which value occurs) and a column for each of the
    timeseries parameters

    Parameters
    ----------
    data
        The data to have its format changed

    Returns
    -------
    pandas.DataFrame
        Data, now as a tsfresh flat dataset

    Raises
    ------
    ValueError
        If data is not a tslearn timeseries dataset,
        tsfresh stacked dataset or tsfresh flat dataset
    """
    try:
        is_flat_dataset(data)  # will raise TimekeepCheckError if not
        return data
    except TimekeepCheckError:
        pass

    try:
        is_stacked_dataset(data)  # will raise TimekeepCheckError if not

        # Get the id and time values for one "kind" of values
        flat_data = data.loc[
            data["kind"] == data.loc[0, "kind"], ["id", "time"]
        ].reset_index(drop=True)

        # Add the values as columns
        data_value_dtype = data["value"].dtype
        for col_name in np.unique(data["kind"]):
            flat_data[col_name] = data.loc[
                data.loc[:, "kind"] == col_name, "value"
            ].astype(data_value_dtype)

        return flat_data
    except TimekeepCheckError:
        pass

    try:
        is_timeseries_dataset(data)  # will raise TimekeepCheckError if not
        n, t, d = data.shape
        id_ = np.tile(np.arange(n), t)
        time_ = np.tile(np.arange(t), n)
        values_ = data.reshape(n * t, d)  # check if this reshape is correct
        df = pd.DataFrame({"id": id_, "time": time_})
        for value in range(d):
            df[str(value)] = values_[:, value]

        return df
    except TimekeepCheckError:
        pass

    raise ValueError(
        "Did not recognise data of type {}. Cannot convert to flat dataset".format(
            type(data)
        )
    )


def to_stacked_dataset(data):
    """
    Convert a tslearn timeseries or tsfresh flat dataset
    to a tsfresh stacked dataset

    A stacked dataset is a DataFrame with columns for 'id' (of timeseries),
    'time' (at which value occurs), 'kind' (of value),
    and 'value' (of timeseries parameter)

    Parameters
    ----------
    data
        The data to have its format changed

    Returns
    -------
    pandas.DataFrame
        Data, now as a tsfresh stacked dataset

    Raises
    ------
    ValueError
        If data is not a tslearn timeseries dataset,
        tsfresh stacked dataset or tsfresh flat dataset
    """
    try:
        is_flat_dataset(data)
        d = data.shape[1] - 2
        id_ = np.tile(data["id"].to_numpy(), d)
        time_ = np.tile(data["time"].to_numpy(), d)
        kind_ = np.repeat(
            np.array([col for col in data.columns if col not in ("time", "id")]),
            data.shape[0],
        )
        values_ = (
            data[[col for col in data.columns if col not in ("time", "id")]]
            .to_numpy()
            .flatten("F")  # flatten Fortran (column-major) order
        )

        return pd.DataFrame({"id": id_, "time": time_, "kind": kind_, "value": values_})
    except TimekeepCheckError:
        pass

    try:
        is_stacked_dataset(data)
        return data
    except TimekeepCheckError:
        pass

    try:
        is_timeseries_dataset(data)  # will raise TimekeepCheckError if not
        n, t, d = data.shape
        id_ = np.tile(np.arange(n), t * d)
        time_ = np.tile(np.arange(t), n * d)
        kind_ = np.repeat(np.arange(d), n * t)
        values_ = data.flatten()  # check if this reshape is correct
        return pd.DataFrame({"id": id_, "time": time_, "kind": kind_, "value": values_})

    except TimekeepCheckError:
        pass

    raise ValueError(
        "Did not recognise data of type {}. Cannot convert to stacked dataset".format(
            type(data)
        )
    )
