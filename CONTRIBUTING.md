# Contributing

Thank you for contributing to `timekeep`. Feel free to open an issue reporting a bug
or requesting a new feature. If you would like to open a pull request, please read
the information below.

### Code quality
`timekeep` uses [`black`][black] to format code and [`isort`][isort] to maintain clean
importing. Make sure you run these packages on your code changes otherwise the tests will fail.

### Tests
[travis][travis] is used for continuous integration. If you're fixing a bug, please add a [unit test][tests]
which tests for the bug.

[black]: https://github.com/psf/black
[isort]: https://github.com/search?q=isort
[travis]: https://travis-ci.com/

[tests]: tests/