from setuptools import setup

import timekeep

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="timekeep",
    version=timekeep.__version__,
    description="Defensive timeseries analytics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="timeseries time series analysis data science",
    url="https://github.com/TTitcombe/timekeep",
    author="Tom Titcombe",
    author_email="t.j.titcombe@gmail.com",
    license="MIT",
    packages=["timekeep"],
    install_requires=["numpy", "pandas"],
    include_package_data=True,
    python_requires=">=3.6",
)
