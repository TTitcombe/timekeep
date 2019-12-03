from setuptools import setup

import timekeep

setup(
    name="timekeep",
    version=timekeep.__version__,
    description="Defensive timeseries analytics",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="timeseries time series analysis data science",
    url="https://github.com/TTitcombe/timekeep",
    author="Tom Titcombe",
    author_email="t.j.titcombe@gmail.com",
    license="MIT",
    packages=["timekeep"],
    install_requires=["numpy", "sklearn"],
    include_package_data=True,
)
