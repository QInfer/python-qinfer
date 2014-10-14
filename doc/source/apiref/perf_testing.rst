..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _perf_testing:
    
.. currentmodule:: qinfer

Performance Testing
===================

Introduction
------------

.. todo::

	Write guide on this.

These functions provide performance testing support, allowing for the quick
comparison of models, experiment design heuristics and quality parameters.
QInfer's performance testing functionality can be quickly applied without
writing lots of boilerplate code every time. For instance::

	>>> import qinfer
	>>> n_particles = int(1e5)
	>>> perf = qinfer.perf_test(
	...     qinfer.SimplePrecessionModel(), n_particles,
	...     qinfer.UniformDistribution([0, 1]), 200,
	...     qinfer.ExpSparseHeuristic
	... )

Function Reference
------------------

.. autofunction:: perf_test
