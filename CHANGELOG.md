# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added

### Changed
- Check failures raise a custom error, not `AssertionError`

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

[Unreleased]: https://github.com/TTitcombe/timekeep/compare/0.1...HEAD
[0.1.0]: https://github.com/TTitcombe/timekeep/releases/tag/0.1

[sklearn]: https://scikit-learn.org
[tsfresh]: https://tsfresh.readthedocs.io
[tslearn]: https://tslearn.readthedocs.io