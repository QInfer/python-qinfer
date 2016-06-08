..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _parallel_guide:
    
.. currentmodule:: qinfer

Parallel Execution of Models
============================

Introduction
------------

**QInfer** provides tools to expedite simulation by distributing computation
across multiple nodes using standard parallelization tools.

Distributed Computation with IPython
------------------------------------

The `ipyparallel`_ package (previously ``IPython.parallel``) provides
facilities for parallelizing computation across multiple cores and/or nodes.
`ipyparallel`_ separates computation into a *controller* that is responsible
for one or more *engines*, and a *client* that sends commands to these engines
via the controller. **QInfer** can use a client to send likelihood evaluation
calls to engines, via the :class:`DirectViewParallelizedModel` class.

This class takes a :class:`~ipyparallel.DirectView` onto one
or more engines, typically obtained with an expression similar to
``client[:]``, and splits calls to :meth:`~qinfer.Model.likelihood`
across the engines accessible from the :class:`~ipyparallel.DirectView`.

.. note::

    **QInfer** does not include `ipyparallel`_ in its installation, so you must
    `install <https://ipyparallel.readthedocs.io/en/latest/#installing-ipython-parallel>`_
    it separately. To run the example code also requires some initialization,
    which is also described in the
    `docs <https://ipyparallel.readthedocs.io/en/latest/intro.html#getting-started>`_.

>>> from ipyparallel import Client # doctest: +SKIP
>>> from qinfer import SimplePrecessionModel # doctest: +SKIP
>>> from qinfer import DirectViewParallelizedModel # doctest: +SKIP
>>> c = Client() # doctest: +SKIP
>>> serial_model = SimplePrecessionModel() # doctest: +SKIP
>>> parallel_model = DirectViewParallelizedModel(serial_model, c[:]) # doctest: +SKIP

The newly decorated model will now distribute likelihood calls, such that each
engine computes the likelihood for an equal number of particles. As a
consequence, information shared per-experiment or per-outcome is local to each
engine, and is not distributed. Therefore, this approach works best at quickly
parallelizing where the per-model cost is significantly larger than the
per-experiment or per-outcome cost.

.. note::

    The :class:`DirectViewParallelizedModel` assumes that it has ownership
    over engines, such that the behavior is unpredictable if any further
    commands are sent to the engines from outside the class.

Distributed Performance Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As an alternative to distributing a single likelihood call across multiple
engines, **QInfer** also supports distributed :ref:`perf_testing_guide`. Under this
model, each engine performs an independent trial of an estimation procedure,
which is then collected by the client process. Distributed performance testing
is implemented using the :func:`~qinfer.perf_test_multiple` function, with the
keyword argument ``apply`` provided. For instance, the `ipyparallel`_ package
offers a :class:`~ipyparallel.LoadBalancedView` class whose
:meth:`~ipyparallel.LoadBalancedView.apply` method sends tasks to engines
according to their respective loads.

>>> lbview = client.load_balanced_view() # doctest: +SKIP
>>> performance = qi.perf_test_multiple(
...     100, serial_model, 6000, prior, 200, heuristic_class,
...     apply=lbview.apply
... ) # doctest: +SKIP

Examples of both approaches to parallelization are provided as a
`Jupyter Notebook <http://nbviewer.jupyter.org/github/QInfer/python-qinfer/blob/master/examples/parallelization.ipynb>`_.

GPGPU-based Likelihood Computation with PyOpenCL
------------------------------------------------

Though **QInfer** does not yet have built-in support for GPU-based
parallelization, `PyOpenCL`_ can be used to effectively distribute models as
well. Here, the Cartesian product over outcomes, models and experiments matches
closely the OpenCL concept of a *global ID*, as
`this example <https://gist.github.com/cgranade/6137168>`_ demonstrates.
Once a kernel is developed in this way, PyOpenCL will allow for it to be used
with any available OpenCL-compliant device.

Note that for sufficiently fast models, the overhead of copying data between
the CPU and GPU may overwhelm any speed benefits obtained by this
parallelization.

.. _ipyparallel: https://ipyparallel.readthedocs.org/en/latest/
.. _PyOpenCL: http://documen.tician.de/pyopencl/
