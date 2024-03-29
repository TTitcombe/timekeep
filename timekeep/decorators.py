"""
Decorators for checking data structure
"""
import timekeep.checks as tkc


def is_timeseries_dataset(func):
    def inner(*args, **kwargs):
        data = func(*args, **kwargs)
        tkc.is_timeseries_dataset(data)
        return data

    return inner


def is_flat_dataset(func):
    def inner(*args, **kwargs):
        data = func(*args, **kwargs)
        tkc.is_flat_dataset(data)
        return data

    return inner


def is_stacked_dataset(func):
    def inner(*args, **kwargs):
        data = func(*args, **kwargs)
        tkc.is_stacked_dataset(data)
        return data

    return inner


def is_sklearn_dataset(func):
    def inner(*args, **kwargs):
        data = func(*args, **kwargs)
        tkc.is_sklearn_dataset(data)
        return data

    return inner


def is_shape(shape):
    def is_shape_decorator(func):
        def inner(*args, **kwargs):
            data = func(*args, **kwargs)
            tkc.is_shape(data, shape)
            return data

        return inner

    return is_shape_decorator


def none_missing(func):
    def inner(*args, **kwargs):
        data = func(*args, **kwargs)
        tkc.none_missing(data)
        return data

    return inner


def full_timeseries(empty_value=0.0):
    def full_timeseries_decorator(func):
        def inner(*args, **kwargs):
            data = func(*args, **kwargs)
            tkc.full_timeseries(data, empty_value=empty_value)
            return data

        return inner

    return full_timeseries_decorator


def at_least_n_datapoints(n):
    def at_least_n_decorator(func):
        def inner(*args, **kwargs):
            data = func(*args, **kwargs)
            tkc.at_least_n_datapoints(data, n)
            return data

        return inner

    return at_least_n_decorator


def fewer_than_n_datapoints(n):
    def fewer_than_n_decorator(func):
        def inner(*args, **kwargs):
            data = func(*args, **kwargs)
            tkc.fewer_than_n_datapoints(data, n)
            return data

        return inner

    return fewer_than_n_decorator


def datapoints_in_range(n_lower, n_upper):
    def datapoints_in_range_decorator(func):
        def inner(*args, **kwargs):
            data = func(*args, **kwargs)
            tkc.at_least_n_datapoints(data, n_lower)
            tkc.fewer_than_n_datapoints(data, n_upper)
            return data

        return inner

    return datapoints_in_range_decorator


def has_uniform_length(func):
    def inner(*args, **kwargs):
        data = func(*args, **kwargs)
        tkc.has_uniform_length(data)
        return data

    return inner
