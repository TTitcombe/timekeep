"""
Functions for converting between data formats
"""
import numpy as np
from sklearn.base import BaseEstimator
from tslearn.utils import to_sklearn_dataset


def convert_timeseries_input(func):
    def inner(*args, **kwargs):
        x = args[1]  # self, then X
        assert isinstance(x, np.ndarray)

        x = to_sklearn_dataset(x)
        args = [args[i] if i != 1 else x for i in range(len(args))]

        return func(*args, **kwargs)

    return inner


def accept_timeseries_input(cls: BaseEstimator) -> BaseEstimator:
    """
    A class decorator which alters sklearn BaseEstimator classes
    to accept timeseries datasets as input.
    """
    cls.fit = convert_timeseries_input(cls.fit)
    cls.transform = convert_timeseries_input(cls.transform)
    cls.fit_transform = convert_timeseries_input(cls.fit_transform)
    cls.predict = convert_timeseries_input(cls.predict)

    return cls
