..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _simple_est_guide:
    
.. currentmodule:: qinfer

Simple Estimation Functions
===========================

**QInfer** provides several functions to help you get up and
running quickly with common estimation tasks, without having to
worry about explicitly specifying models and distributions. Later,
we'll see how to build up custom estimation problems in a
straightforward and structured manner.
For now, though, let's start by diving into how to use **QInfer**
to learn a single precession frequency.

In particular, suppose that you have a qubit that starts in
the :math:`\ket{+} = (\ket{0} + \ket{1}) / \sqrt{2}` state, then
evolves under :math:`U(t) = \exp(-\ii \omega \sigma_z)` for an
unknown frequency :math:`\omega`. Then measuring the qubit in the
:math:`\sigma_x` basis results in observing a :math:`1` with
probability :math:`\sin^2(\omega t / 2)`. We can estimate
the precession frequency
:math:`\omega` with **QInfer** using the :func:`~qinfer.simple_est_prec`
function.

As an example, let's consider an experiment for learning :math:`\omega`
that consists of taking 40 measurements at each time
:math:`t_k = k / (2 \omega_{\max})`, where :math:`k = 0, \dots, N - 1`
indexes each measurement, :math:`\omega_{\max}` is the maximum plausible
frequency to be estimated, and where :math:`N` is the number of
distinct times measured. We can generate this data using
:func:`~numpy.random.binomial`:

>>> omega_max = 100
>>> true_omega = 70.3
>>> ts = np.arange(1, 51) / (2 * omega_max)
>>> counts = np.random.binomial(40, p=np.sin(true_omega * ts / 2) ** 2)

We pass this data to **QInfer** as an array with three *columns* (that is,
shape ``(50, 3)`` for this example), corresponding respectively to
the observed counts, the time at which the counts were observed, and
the number of measurements taken at that time.

>>> data = np.column_stack([counts, ts, np.ones_like(counts) * 40])

Finally, we're ready to call :func:`~qinfer.simple_est_prec`:

>>> from qinfer import simple_est_prec
>>> mean, cov = simple_est_prec(data, freq_max=omega_max)

The returned ``mean`` and ``cov`` tell us the mean and covariance,
respectively, resulting from the frequency estimation problem.

>>> print(mean) # doctest: +SKIP
70.3822376258

Data can also be passed to :func:`~qinfer.simple_est_prec` as
a string containing the name of a CSV-formatted data file, or
as a `Pandas`_ :class:`~pandas.DataFrame`. The latter is especially
useful for loading data from formats such as Excel spreadsheets,
using :func:`~pandas.read_excel`.

For more information, please see the :ref:`API reference <simple_est>`
or the examples below.

Related Examples
----------------

- :example_nb:`simple_precession_example`
- :example_nb:`randomized_benchmarking`


.. _Pandas: http://pandas.pydata.org/