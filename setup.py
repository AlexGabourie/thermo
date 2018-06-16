#!/usr/bin/env python

from distutils.core import setup
import setuptools

setup(name='thermo',
      version='1.0.1',
      description='MD thermal properties functions',
      author='Alexander Gabourie',
      author_email='gabourie@stanford.edu',
      packages=setuptools.find_packages(),
      install_requires=['matplotlib',
                        'pyfftw',
                        'scipy',
                        'ase',
                        'atomman==1.1.5'],
     )
