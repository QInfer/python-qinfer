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
metric collected by **QInfer** about the performance.
In addition to these fields, each field in ``model.expparams_dtype``
is added as a field to the performance results structure to record
what measurements are performed.

For a single performance trial, the shape of the performance results
array is ``(n_exp, )``, such that ``perf[idx_exp]``
returns metrics describing the performance immediately following collecting
the datum ``idx_exp``. Some fields are not scalar-valued, such that ``perf[field]``
then has shape ``(n_exp, ) + field_shape``.

On the other hand, when multiple trials are collected by ``perf_test_multiple``, the results are
returned as an array with the same fields, but with an additional index over trials,
for a shape of ``(n_trials, n_exp)``.

+--------------------+---------+--------------------------------------------------+
| Field              | Type    | Shape                                            |
+====================+=========+==================================================+
| ``elapsed_time``   | `float` | scalar                                           |
+--------------------+---------+--------------------------------------------------+
| Time (in seconds) elapsed during the SMC update for this                        |
| experiment. Includes resampling, but excludes experiment                        |
| design, generation of "true" data and calculation of                            |
| performance metrics.                                                            |
+--------------------+---------+--------------------------------------------------+
| ``loss``           | `float` | scalar                                           |
+--------------------+---------+--------------------------------------------------+
| Decision-theoretic loss incured by the estimate after updating                  |
| with this experiment, given by the quadratic loss                               |
| :math:`\Tr(Q (\hat{\vec{x}} -\vec{x}) (\hat{\vec{x}} - \vec{x})^{\mathrm{T}})`. |
| If the true and estimation models have                                          |
| different numbers of parameters, the loss will only be                          |
| evaluated for those parameters that are in common (aligning the                 |
| two vectors at the right).                                                      |  
+--------------------+---------+--------------------------------------------------+
| ``resample_count`` | `int`   | scalar                                           |
+--------------------+---------+--------------------------------------------------+
| Number of times that resampling was performed on the SMC                        |
| updater.                                                                        |
+--------------------+---------+--------------------------------------------------+
| ``outcome``        | `int`   | scalar                                           |
+--------------------+---------+--------------------------------------------------+
| Outcome of the experiment that was performed.                                   |
+--------------------+---------+--------------------------------------------------+
| ``true``           | `float` | ``(true_model.n_modelparams, )``                 |
+--------------------+---------+--------------------------------------------------+
| Vector of model parameters used to simulate data. For                           |
| time-dependent models, this changes with each experiment as per                 |
| ``true_model.update_timestep``.                                                 |
+--------------------+---------+--------------------------------------------------+
| ``est``            | `float` | ``(model.n_modelparams, )``                      | 
+--------------------+---------+--------------------------------------------------+
| Mean vector of model parameters over the current posterior.                     |
+--------------------+---------+--------------------------------------------------+

.. _record array: http://docs.scipy.org/doc/numpy/user/basics.rec.html
