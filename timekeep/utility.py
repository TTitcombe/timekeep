"""
Functions for managing timeseries
"""
import numpy as np


def find_stop_indices(data: np.ndarray, empty_value: int = 0.0) -> np.ndarray:
    """
    Find the indices in the time dimension at which each timeseries in data
    ends. Default empty value is 0.0
    """
    n, t, d = data.shape
    end_indices = np.empty((n,))
    for timeseries_num in range(n):
        timeseries_end = t
        for dim_num in range(d):
            timeseries = data[timeseries_num, :, dim_num]
            end_idx = t
            for idx in np.arange(t)[timeseries == empty_value][::-1]:
                if idx == end_idx - 1:
                    end_idx = idx
                else:
                    break
        if end_idx < timeseries_end:
            timeseries_end = end_idx
        end_indices[timeseries_num] = timeseries_end

    return end_indices


def uniform_timeseries_length(data: np.ndarray, empty_value: int = 0.0) -> bool:
    """
    Identify whether or not the timeseries stop at the same time index.
    """
    end_indices = find_stop_indices(data, empty_value=empty_value)
    return np.unique(end_indices).size == 1
