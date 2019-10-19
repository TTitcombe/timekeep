timekeep
--------

timekeep is a package for defensive timeseries analysis. The package provides decorators for
lightweight confirmation of your data assumptions and seamless transition between timeseries and
other types of data.

To check your data assumptions, simply do::

    >>> import timekeep.decorators as tkd
    >>> @tkd.is_shape((-1, 100, 2))
    >>> @tkd.none_missing
    >>> def load_my_data():
    >>>     # Your data loading function
    >>>     pass

If the data returned from load_my_data either has NaNs or does not have the shape
(X, 100, 2), an AssertionError will be raised. timekeep's decorators stop you
from having to add messy assertions into your code.

To interface timeseries data with, for example, sklearn classes, do::

    >>> from sklearn.decomposition import PCA
    >>> import timekeep.conversion as tkc
    >>> @tkc.timeseries_transformer
    >>> class TimeseriesPCA(PCA):
    >>>     pass

TimeseriesPCA will accept timeseries data and convert it to sklearn-shaped data
before running the code.