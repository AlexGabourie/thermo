#!/usr/bin/env python

from distutils.core import setup
import setuptools

setup(name='thermo',
      version='0.3',
      description='MD thermal properties functions',
      author='Alexander Gabourie',
      author_email='gabourie@stanford.edu',
      packages=setuptools.find_packages(),
      include_package_data=True,
      install_requires=['matplotlib',
                        'pyfftw',
                        'scipy',
                        'ase>=3.20.1',
                        'atomman==1.2.3'],
     )
