"""
Decorators for checking data structure
"""
import timekeep.checks as tkc
from tslearn.utils import to_sklearn_dataset


def is_timeseries(func):
    def inner(*args, **kwargs):
        data = func(*args, **kwargs)
        result = tkc.is_timeseries(data)
        if not result:
            raise RuntimeError("Is not timeseries")
        # TODO what to do with result
        return data

    return inner


def is_shape(shape):
    def is_shape_decorator(func):
        def inner(*args, **kwargs):
            data = func(*args, **kwargs)
            result = tkc.is_shape(data, shape)
            if not result:
                raise RuntimeError("Shapes do not match")
            # TODO what to do with result
            return data

        return inner

    return is_shape_decorator


def convert_timeseries_input(func):
    def inner(*args, **kwargs):
        x = args[1]  # self, then X
        x = to_sklearn_dataset(x)
        args = [args[i] if i != 1 else x for i in range(len(args))]

        return func(*args, **kwargs)

    return inner


def accept_timeseries_input(cls):
    """
    A class decorator which alters sklearn BaseEstimator classes
    to accept timeseries datasets as input.
    """
    cls.fit = convert_timeseries_input(cls.fit)
    cls.transform = convert_timeseries_input(cls.transform)
    cls.fit_transform = convert_timeseries_input(cls.fit_transform)
    cls.predict = convert_timeseries_input(cls.predict)

    return cls
