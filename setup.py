#!/usr/bin/env python

from distutils.core import setup

setup(name='thermo',
      version='1.0.1',
      description='MD thermal properties functions',
      author='Alexander Gabourie',
      author_email='gabourie@stanford.edu',
      packages=['thermo'],
      install_requires=['matplotlib', 'pyfftw', 'scipy'],
     )
