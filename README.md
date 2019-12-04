# timekeep
[![Build Status](https://travis-ci.com/TTitcombe/timekeep.svg?branch=master)](https://travis-ci.com/TTitcombe/timekeep)
[![codecov](https://codecov.io/gh/TTitcombe/timekeep/branch/master/graph/badge.svg)](https://codecov.io/gh/TTitcombe/timekeep)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TTitcombe/timekeep/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A python package for defensive timeseries analytics.

## What is `timekeep`
All code needs to make assumptions about the data it uses: its shape, content, or format.
But constantly checking and re-checking the assumptions can make the code unwieldy.
Timeseries data presents additional complications in that many common python packages, such as
[`pandas`][pandas] and [`scikitlearn`][sklearn], expect 2-dimensional, static data.

`timekeep` protects your timeseries data by providing simple decorators to check those assumptions.
`timekeep` is heavily inspired by [`engarde`][engarde], but does not share its assumption about
data being a [`pandas`][pandas] dataframe.

## Quickstart
### How to install
`timekeep` is available from [PyPi][pypi]. Run
```bash
pip install timekeep
```
to install.

**Important**:
`timekeep` is currently on version `0.x.y` and is in active development. As the version reflects,
the codebase is liable to change. Once main functionality has been applied, a stable version `1.0.0`
will be released.

### How to use
`timekeep` provides decorators to be used on functions which return a timeseries dataset.
Each decorator checks an assumption about the data and raises an error if this assumption is not met.

```python
import numpy as np
import timekeep.decorators as tkd

# Check that data returned from wont_raise
# has shape (-1, 10, 2)
@tkd.is_shape((-1, 10, 2))
def wont_raise():
  return np.random.random((10, 10, 2))

# Check that data returned from will_raise is a
# timeseries dataset (has three dimensions)
@tkd.is_timeseries
def will_raise():
  return np.random.random((10, 2))
```

Another key feature of `timekeep` is conversion between timeseries data formats; `timekeep` can convert between
[`tslearn`][tslearn] style timeseries dataset and [`sklearn`][sklearn] datasets, with support for [`tsfresh`][tsfresh]
datasets in progress. The conversion functions can be applied as decorators to methods and functions to automatically
convert your data as you supply it. For example:

```python
import numpy as np
import timekeep.conversion as tkc
from sklearn.preprocessing import PCA

# Convert timeseries dataset to sklearn for input into sklearn.preprocessing.PCA
@tkc.convert_timeseries_input
def run_pca_on_timeseries(data):
  return PCA().fit(data)

timeseries_data = np.random.random((10, 100, 2))
run_pca_on_timeseries(timeseries_data)
```

See the [`examples`][examples] folder for more. You can launch this repo on [`binder`][binder_timekeep] 
and run the examples without any installation.

## Contributing
Any and all help welcome. Please see the [contributing guide][contributing].

[binder_timekeep]: https://mybinder.org/v2/gh/TTitcombe/timekeep/master
[engarde]: https://github.com/engarde-dev/engarde
[pandas]: https://pandas.pydata.org/
[pypi]: https://pypi.org/project/timekeep/0.1/
[sklearn]: https://scikit-learn.org/stable/index.html
[tsfresh]: https://tsfresh.readthedocs.io
[tslearn]: https://tslearn.readthedocs.io

[contributing]: CONTRIBUTING.md
[examples]: examples/