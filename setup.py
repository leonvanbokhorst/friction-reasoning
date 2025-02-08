"""Setup file for friction-reasoning-data-gen package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="friction-reasoning-data-gen",
    version="0.1.0",
    author="Leon van Bokhorst",
    author_email="mail@leonvanbokhorst.com",
    description="A dataset generator for multi-agent reasoning with designed friction points",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leonvanbokhorst/friction-reasoning-data-gen",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
) 