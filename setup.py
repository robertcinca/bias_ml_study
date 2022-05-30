# setup.py
from setuptools import find_packages, setup

setup(
    setup_requires=["pbr"],
    pbr=True,
    name="ml_bias_explainability",
    packages=find_packages(),
    install_requires=[],
)
