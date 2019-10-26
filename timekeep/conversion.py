"""
Functions for converting between data formats
"""
import numpy as np
from sklearn.base import TransformerMixin
from tslearn.utils import to_sklearn_dataset, to_time_series_dataset


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
        assert len(data.shape) == 2
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
