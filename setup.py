#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages
from distutils.core import setup
from projecteuler import __author__, __version__, __license__

setup(
        name = "ProjectEuler-module",
        version = __version__,
        description = "Project Euler module",
        license = __license__,
        author = __author__,
        url = "https://github.com/kinketu/ProjectEuler-module.git",
        keywords = "Project Euler module",
        packages = find_packages(),
        install_requires = [],
        )
