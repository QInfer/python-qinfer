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

.. autofunction:: perf_test_multiple

.. _perf_testing_struct:

Performance Results Structure
-----------------------------

Perfromance results, as collected by :func:`perf_test`, are returned as
a `record array`_ with several fields, each describing a different
metric collected by QInfer about the performance. For a single performance
trial, the shape of this array is ``(n_exp, )``, such that ``perf[idx_exp]``
returns metrics describing the performance immediately following collecting
the datum ``idx_exp``.

+----------------+-------+----------------------------------------------------+
| Field          | Type  | Description                                        |
+================+=======+====================================================+
| elapsed_time   | float | Time (in seconds) elapsed during                   |
|                |       | the SMC update for this experiment.                |
|                |       | Includes resampling, but excludes experiment       |
|                |       | design, generation of "true" data and calculation  |
|                |       | of performance metrics.                            |
+----------------+-------+----------------------------------------------------+
| loss           | float | Decision-theoretic loss incured by the estimate    |
|                |       | after updating with this experiment, given by the  |
|                |       | quadratic loss :math:`\Tr(Q (\hat{\vec{x}} -       |
|                |       | \vec{x}) (\hat{\vec{x}} - \vec{x})^{\mathrm{T}})`. |
+----------------+-------+----------------------------------------------------+
| resample_count | int   | Number of times that resampling was performed on   |
|                |       | the SMC updater.                                   |
+----------------+-------+----------------------------------------------------+

.. _record array: http://docs.scipy.org/doc/numpy/user/basics.rec.html
