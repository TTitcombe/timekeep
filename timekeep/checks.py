"""
Checks to perform on data
"""


def is_timeseries(data):
    """
    Check if data is in a timeseries format.
    Timeseries data has 3 dimensions:
      N - number of data points
      T - number of time points
      D - number of data dimensions
    """
    return len(data.shape) == 3


def is_shape(data, shape):
    """
    Compare the shape of a dataset
    """
    if len(data.shape) != len(shape):
        return False

    shape_comparison = [
        True if dim2 == -1 else dim1 == dim2 for dim1, dim2 in zip(data.shape, shape)
    ]
    return all(shape_comparison)
