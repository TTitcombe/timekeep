"""
Functions for converting between data formats
"""
import numpy as np
from sklearn.base import TransformerMixin
from tslearn.utils import to_sklearn_dataset, to_time_series_dataset


def convert_timeseries_input(func):
    def inner(*args, **kwargs):
        assert len(args) >= 2
        x = args[1]  # self, then X
        assert isinstance(x, np.ndarray)

        x = to_sklearn_dataset(x)
        args = [args[i] if i != 1 else x for i in range(len(args))]

        return func(*args, **kwargs)

    return inner


def convert_output_to_timeseries(func):
    def inner(*args, **kwargs):
        data = func(*args, **kwargs)
        assert len(data.shape) == 2

        return to_time_series_dataset(data)

    return inner


def timeseries_transformer(cls: TransformerMixin) -> TransformerMixin:
    """
    A class decorator which alters sklearn Transformer classes
    to accept timeseries datasets as input.
    """
    cls.fit = convert_timeseries_input(cls.fit)
    cls.transform = convert_timeseries_input(cls.transform)
    cls.fit_transform = convert_timeseries_input(cls.fit_transform)

    return cls
