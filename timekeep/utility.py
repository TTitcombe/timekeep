"""
Functions for managing timeseries
"""
import numpy as np


def get_timeseries_lengths(data: np.ndarray, empty_value=np.nan) -> np.ndarray:
    """
    Find the indices in the time dimension at which each timeseries in data
    ends.

    Empty values in the data are assumed to be np.nan by default

    Parameters
    ----------
    data : np.ndarray
        The data to check
    empty_value
        The value which denotes empty data. np.nan by default.

    Returns
    -------
    np.ndarray
        An array containing the length of each timeseries in data
    """
    n, t, d = data.shape
    end_indices = np.empty((n,))
    for timeseries_num in range(n):
        timeseries_end = t

        timeseries = data[timeseries_num]

        if np.isnan(empty_value):
            idxs = np.arange(t)[np.isnan(timeseries).all(axis=1)]
        else:
            idxs = np.arange(t)[(timeseries == empty_value).all(axis=1)]

        end_idx = timeseries_end
        for idx in idxs[::-1]:
            if idx == end_idx - 1:
                end_idx = idx
            else:
                break

        if end_idx < timeseries_end:
            timeseries_end = end_idx
        end_indices[timeseries_num] = timeseries_end

    return end_indices
