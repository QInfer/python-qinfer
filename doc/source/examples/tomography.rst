..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _tomography_example:
    
Single-Qubit State Tomography
=============================

TODO: describe this example.

Command-Line Options
--------------------

.. program:: python -m qinfer.examples.qubit_tomography_example

Basic Options
~~~~~~~~~~~~~

.. option:: -n NP, --n_particles=NP

    Specifies how many particles to use in the SMC
    approximation. [default: 5000]
    
.. option:: -e NE, --n_exp=NE

    Specifies how many measurements are to be made. [default: 100]
    
.. option:: -a ALGO, --algorithm=ALGO

    Specifies which algorithm to use; currently 'SMC' and 'SMC-ABC' are
    supported. [default: SMC]
    
Resampling Options
~~~~~~~~~~~~~~~~~~
    
.. option:: -r ALGO, --resampler=ALGO

    Specifies which resampling algorithm to use; currently 'LW',
    'DBSCAN-LW' and 'WDBSCAN-LW' are supported. [default: LW]
    
.. option:: --lw-a=A

    Parameter :math:`a` of the [LW01]_ resampling algorithm. [default: 0.98]
    
.. option:: --dbscan-eps=EPS

    Epsilon parameter for the DBSCAN-based resamplers. [default: 0.5]
    
.. option:: --dbscan-minparticles=N

    Minimum number of particles allowed in a cluster by the DBSCAN-based resamplers.
    [default: 5]
    
.. option:: --wdbscan-pow=POW

    Power by which the weight is to be raised in the WDBSCAN weighting step.
    [default: 0.5]
    
ABC Options
~~~~~~~~~~~
    
.. option:: --abctol=TOL

    Specifies the tolerance used by the SMC-ABC algorithm. [default: 8e-6]
    
.. option:: --abcsim=SIM

    Specifies how many simulations are used by each ABC step. [default: 10000]

Plotting Options
~~~~~~~~~~~~~~~~

.. option:: -p, --plot

    Enables plotting of data from this example.
                                
Debugging Options
~~~~~~~~~~~~~~~~~

.. option:: -v, --verbose
    
    Prints additional debugging information.

