"""Shim for editable installs on pip < 21.3, which predates PEP 660.

All project metadata lives in pyproject.toml.
"""

from setuptools import setup

setup()
