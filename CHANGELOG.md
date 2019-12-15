# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Changed
- Renamed `utility.find_stop_indices` to `utility.get_timeseries_lengths`
- Renamed `checks.check_uniform_length` to `checks.has_uniform_length`
- Renamed `decorators.uniform_length` to `decorators.has_uniform_length`

### Removed
- `utility.uniform_timeseries_length`
- [`sklearn`][sklearn] is no longer a dependency

### Fixed
- Bugs in `conversion.to_timeseries_dataset` and `conversion.to_sklearn_dataset`
  conversion format

## [0.2.0] - 2019-12-04
### Added
- Checks for dataset format
    - [`tsfresh`][tsfresh]-style flat and stacked datasets, [`tslearn`][tslearn] timeseries datasets
    and [`sklearn`][sklearn] datasets
- Functions for converting between those formats
- `timekeep` is now tested on Python `3.6`, `3.7`, `3.8`

### Changed
- Check failures raise a custom error, not `AssertionError`

### Removed
- [`tslearn`][tslearn] is no longer a dependency

### Fixed
- Conversion decorators work on functions and methods

## [0.1.0] - 2019-10-25
### Added
- First release of `timekeep`
- Conversion decorators from [`tslearn`][tslearn] timeseries datasets to [`sklearn`][sklearn] datasets
- Checks for:
    - Missing data
    - Data shape
    - Uniform timeseries length
    - Non-full timeseries
- Decorators for all checks

[Unreleased]: https://github.com/TTitcombe/timekeep/compare/0.2.0...HEAD
[0.1.0]: https://github.com/TTitcombe/timekeep/releases/tag/0.1
[0.2.0]: https://github.com/TTitcombe/timekeep/releases/tag/0.2.0

[sklearn]: https://scikit-learn.org
[tsfresh]: https://tsfresh.readthedocs.io
[tslearn]: https://tslearn.readthedocs.io
