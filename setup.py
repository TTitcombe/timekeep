from setuptools import setup

setup(
    name="timekeep",
    version="0.1",
    description="Defensive timeseries analytics",
    author="Tom Titcombe",
    author_email="t.j.titcombe@gmail.com",
    packages=["timekeep"],
    install_requires=["numpy", "tslearn"],
)
