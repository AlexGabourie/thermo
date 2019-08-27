thermo - A GPUMD Helper Package
===========================================

The thermo, or `thermo`_, package is a set of `Python`_  tools to interface with the molecular dynamics (MD) simulator `GPUMD`_ (and `LAMMPS`_ for comparison) for the purpose of thermodynamic simulations (i.e. thermal conductivity).

Currently, the functionality is limited as it serves specific research purposes at this time; however, the long-term plan is to make this package to primarily serve `GPUMD`_ with only minor supporting functions for `LAMMPS`_ for the purpose of checking force-consistency between the simulators.

The documenation is produced and maintained by `Alex Gabourie <https://github.com/AlexGabourie>`_ at Stanford University. It outlines the structure and usage of the `thermo`_ package.  

Documentation
-------------

| This package contains four subpackages:
| 1. **gpumd** : Python interface specific to `GPUMD`_. 
| 2. **lammps** : Python interface specific to `LAMMPS`_.
| 3. **shared** : Used strictly to compare `GPUMD`_ and `LAMMPS`_.
| 4. **tools** : Extra support for more general MD related content.

..
	Subpackages
	-----------

.. toctree::
	
    thermo.gpumd
    thermo.lammps
    thermo.shared
    thermo.tools

..
	Module contents
	---------------

	.. automodule:: thermo
	    :members:
	    :undoc-members:
	    :show-inheritance:

.. _thermo: https://github.com/AlexGabourie/thermo
.. _GPUMD: https://github.com/brucefan1983/GPUMD
.. _LAMMPS: https://lammps.sandia.gov/
.. _Python: https://www.python.org/


* :ref:`genindex`

.. * :ref:`modindex`
.. * :ref:`search`
