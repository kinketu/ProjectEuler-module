#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages
from distutils.core import setup
from projecteuler import __author__, __version__, __license__, __date__

setup(
        name = "projecteuler",
        version = __version__,
        description = "Project Euler module",
        license = __license__,
        author = __author__,
        url = "https://github.com/kinketu/projecteuler.git",
        keywords = "Project Euler module",
        packages = find_packages(),
        install_requires = [],
        date = __date__
        )
