# timekeep
[![Build Status](https://travis-ci.com/TTitcombe/timekeep.svg?branch=master)](https://travis-ci.com/TTitcombe/timekeep)
[![codecov](https://codecov.io/gh/TTitcombe/timekeep/branch/master/graph/badge.svg)](https://codecov.io/gh/TTitcombe/timekeep)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TTitcombe/timekeep/master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A python package for defensive timeseries analytics.

### Why?
All code needs to make assumptions about the data it uses: its shape, content, or format.
But constantly checking and re-checking the assumptions can make the code unwieldy.

Timeseries data presents additional complications in that many common python packages, such as
[`pandas`][pandas] and [`scikitlearn`][sklearn], expect 2-dimensional, static data.

`timekeep` protects your timeseries data by providing simple decorators to check your assumptions.
Additionally, it provides utility functions for interfacing between timeseries and code which expects
static data.

`timekeep` is heavily inspired by [`engarde`][engarde], but does not share its assumption about
data being a [`pandas`][pandas] dataframe.

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

See the [`examples`][examples] folder for more. You can launch this repo on [`binder`][binder_timekeep] 
and run the examples without any installation.

### How to install
`timekeep` is currently not pip installable. To package a version locally:
1. Clone this repo
2. Navigate to the repo in the command line
3. Run `pip install .`

### Contributing
Any and all help welcome. Please see the [contributing guide][contributing].

[binder]: https://mybinder.org/v2/gh/TTitcombe/timekeep/master
[engarde]: https://github.com/engarde-dev/engarde
[pandas]: https://pandas.pydata.org/
[sklearn]: https://scikit-learn.org/stable/index.html

[contributing]: CONTRIBUTING.md
[examples]: examples/