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

### Pull requests
If you would like to contribute but don't know where to start, check the open issues and pick something you
like the look of.

When working on new code, please fork `timekeep` and make a branch off your fork's master branch.
Push your work to your fork and open a pull request from there.

Use the pull request template where possible. The important things to include are:
1. Which issues, if any, are being fixed by the PR
2. Which tests were added to test the new feature / bug fix
3. How to manually confirm that your code is working as you expect it to

[black]: https://github.com/psf/black
[isort]: https://github.com/search?q=isort
[travis]: https://travis-ci.com/

[tests]: tests/