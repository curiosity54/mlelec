rom setuptools import setup, find_packages

setup(
    name="mlelec",
    version="0.0.0",
    packages=find_packages(include=["mlelec", "mlelec.*"]),
)

