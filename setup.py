from setuptools import setup


def long_desc():
    with open("LONG_DESCRIPTION.rst") as f:
        return f.read()


setup(
    name="timekeep",
    version="0.1",
    description="Defensive timeseries analytics",
    long_description=long_desc(),
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
    install_requires=["numpy", "sklearn", "tslearn"],
    include_package_data=True,
)
