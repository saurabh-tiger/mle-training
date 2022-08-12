from setuptools import find_packages, setup

setup(
    name="houseValuePrediction",
    version="0.0.2",
    description="House Value Prediction",
    author="Saurabh Zinjad",
    author_email="saurabh.zinjad@tigeranalytics.com",
    license="LICENSE.txt",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    long_description=open("README.md").read(),
)
