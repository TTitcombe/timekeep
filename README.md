[![Build Status](https://travis-ci.com/TTitcombe/timekeep.svg?branch=master)](https://travis-ci.com/TTitcombe/timekeep)
[![codecov](https://codecov.io/gh/TTitcombe/timekeep/branch/master/graph/badge.svg)](https://codecov.io/gh/TTitcombe/timekeep)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# timekeep

A python package for defensive timeseries analysis.

### Why?
`timekeep` is heavily inspired by [`engarde`][engarde], a package for ensuring dataframes
take the form you expect. `timekeep` checks your assumptions on timeseries data - data which
has three dimensions.

### How to use
`timekeep` is intended to be used primarily as lightweight decorators which returns a timeseries dataset.
Each decorator checks one assumption about the data and raises an error if this assumption is not met.

```python
import numpy as np

import timekeep.decorators as tkd

@tkd.is_shape((-1, 10, 2))
def wont_raise():
  return np.random.random((10, 10, 2))

@tkd.is_timeseries
def will_raise():
  return np.random.random((10, 2))
```

### Contributing
Any and all help welcome. Please see the [contributing guide][contributing]

[engarde]: https://github.com/engarde-dev/engarde

[contributing]: CONTRIBUTING.md