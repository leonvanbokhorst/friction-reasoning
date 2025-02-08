"""Setup file for friction-reasoning-data-gen package."""

from setuptools import setup, find_packages

setup(
    name="friction-reasoning",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "litellm>=1.0.0",
        "pydantic>=2.0.0",
    ],
    python_requires=">=3.8",
    author="Lonn",
    description="A tool for generating friction-based reasoning data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
) 