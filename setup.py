"""Minimal setup.py for editable install compatibility with older pip/setuptools.

All configuration lives in pyproject.toml. This file allows
`pip install -e .` to work on systems that do not yet support
PEP 660 (editable installs from pyproject.toml only).
"""
from setuptools import setup

if __name__ == "__main__":
    setup()
