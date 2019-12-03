"""
Functions for converting between data formats
"""
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from .checks import (
    is_flat_dataset,
    is_sklearn_dataset,
    is_stacked_dataset,
    is_timeseries_dataset,
)
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
        return to_timeseries_dataset(data)

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
def to_flat_dataset(data) -> pd.DataFrame:
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
        for col_name in np.unique(data["kind"]):
            data_subset = data.loc[
                data.loc[:, "kind"] == col_name, ["id", "time", "value"]
            ].rename(columns={"value": col_name})
            flat_data = flat_data.merge(data_subset, on=["id", "time"])

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


def to_stacked_dataset(data) -> pd.DataFrame:
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


def to_timeseries_dataset(
    data, t: Optional[int] = None, d: Optional[int] = None
) -> np.ndarray:
    """
    Convert a tsfresh or scikit-learn dataset to timeseries dataset.

    A timeseries dataset is a numpy.ndarray with shape (N, T, D).

    Parameters
    ----------
    data
        The data to have its format changed
    t : int, optional
        The number of time points
    d : int, optional
        The number of data parameters

    Returns
    -------
    numpy.ndarray
        Data, now as a timeseries dataset

    Raises
    ------
    ValueError
        If data is not a scikit-learn dataset,
        tsfresh stacked dataset or tsfresh flat dataset
    """
    try:
        is_timeseries_dataset(data)
    except TimekeepCheckError:
        pass
    else:
        return data

    try:
        is_flat_dataset(data)
    except TimekeepCheckError:
        pass
    else:
        d = data.shape[1] - 2

        times = data["time"]
        t = np.max(times) - np.min(times) + 1

        unique_ids = np.unique(data["id"])
        n = unique_ids.size

        ts_data = np.full((n, t, d), np.nan)
        for idx in range(n):
            idx_data = data.loc[data["id"] == unique_ids[idx], :]
            idx_times = idx_data["time"].to_numpy() - np.min(times)
            idx_data = idx_data.drop(["id", "time"], axis=1).to_numpy()

            ts_data[idx, idx_times] = idx_data

        return ts_data

    try:
        is_stacked_dataset(data)
    except TimekeepCheckError:
        pass
    else:
        unique_kinds = np.unique(data["kind"])
        d = len(unique_kinds)

        times = data["time"]
        t = np.max(times) - np.min(times) + 1

        unique_ids = np.unique(data["id"])
        n = unique_ids.size

        stacked_value_dtype = data["value"].to_numpy().dtype
        ts_data = np.full((n, t, d), np.nan, dtype=stacked_value_dtype)
        for idx in range(n):
            for kind_idx in range(d):
                indexes = (data["id"] == unique_ids[idx]) & (
                    data["kind"] == unique_kinds[kind_idx]
                )
                idx_data = data.loc[indexes, "value"].to_numpy()

                idx_times = data.loc[indexes, "time"].to_numpy() - np.min(times)

                ts_data[idx, idx_times, kind_idx] = idx_data

        return ts_data

    try:
        is_sklearn_dataset(data)
    except TimekeepCheckError:
        raise ValueError(
            "Did not recognise data of type {}. Cannot convert to timeseries dataset".format(
                type(data)
            )
        )
    else:
        total_size = data.size
        n = data.shape[0]

        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        if t is None and d is None:
            # assume d = 1
            return np.expand_dims(data, axis=2)
        elif t is None:
            t = int(total_size / (n * d))
        elif d is None:
            d = int(total_size / (n * t))

        return data.reshape((n, t, d))


def to_sklearn_dataset(data) -> np.ndarray:
    """
    Convert a tslearn timeseries or tsfresh dataset
    to a scikit-learn dataset

    A scikit-learn dataset is a numpy.ndarray with 2 axes,
    shape (N, D) where N is number of data points and D is number
    of dimensions.

    Parameters
    ----------
    data
        The data to have its format changed

    Returns
    -------
    numpy.ndarray
        The data, now as a scikit-learn dataset

    Raises
    ------
    ValueError
        If data is not a tslearn timeseries dataset,
        tsfresh stacked dataset or tsfresh flat dataset
    """
    try:
        is_timeseries_dataset(data)
    except TimekeepCheckError:
        pass
    else:
        return data.T.reshape((-1, data.shape[0])).T

    try:
        is_stacked_dataset(data)
    except TimekeepCheckError:
        pass
    else:
        return to_sklearn_dataset(to_timeseries_dataset(data))

    try:
        is_flat_dataset(data)
    except TimekeepCheckError:
        pass
    else:
        return to_sklearn_dataset(to_timeseries_dataset(data))

    try:
        is_sklearn_dataset(data)
    except TimekeepCheckError:
        raise ValueError(
            "Did not recognise data of type {}. Cannot convert to sklearn dataset".format(
                type(data)
            )
        )
    else:
        return data
