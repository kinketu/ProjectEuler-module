#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages
from projecteuler import __author__, __version__, __license__

setup(
        name = "projecteuler",
        version = __version,
        description = "Project Euler module"
        license = __license__,
        author = __author__,
        url = "https://github.com/kinketu/projecteuler.git"
        keywords = "Project Euler module"
        packages = find_packages(),
        install_requires = [],
        )
