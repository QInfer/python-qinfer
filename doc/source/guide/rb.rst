..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _rb_guide:
    
.. currentmodule:: qinfer

Randomized Benchmarking
=======================

Introduction
------------

Randomized benchmarking allows for extracting information about the fidelity
of a quantum operation by exploiting *twirling* errors over an approximate
implementation of the Clifford group [KL+08]_. This provides the advantage that the
fidelity can be learned without simulating the dynamics of a quantum system.
Instead, benchmarking admits an analytic form for the survival probability for
an arbitrary input state in terms of the strength :math:`p` of an equivalent
depolarizing channel.

**QInfer** supports randomized benchmarking by implementing this survival
probability as a *likelihood function*. This allows for randomized benchmarking
to be used together with :ref:`smc_guide`, such that prior information can
be incorporated and robustness to finite sampling can be obtained [GFC14]_.

Regardless of the order or interleaving mode, each model instance for randomized
benchmarking yields 0/1 data, with 1 indicating a survival (measuring the same
state after applying a gate sequence as was initially prepared). To use these
models with data batched over many sequences, model instances can be augmented
by :class:`BinomialModel`.

Zeroth-Order Model
------------------

The :class:`RandomizedBenchmarkingModel` class implements randomized
benchmarking as a **QInfer** model, both in interleaved and non-interleaved
modes. For the non-interleaved mode, there are three model parameters,
:math:`\vec{x} = (p, A_0, B_0)`, given by [MGE12]_ as

.. math::

    A_0 & := \Tr\left[E_\psi \Lambda\left(\rho_\psi - \frac{\ident}{d}\right)\right] \\
    B_0 & := \Tr\left[E_\psi \Lambda\left(\frac{\ident}{d}\right)\right] \\
    p   & := (d F_\ave - 1) / (d - 1),

where :math:`E_\psi` is the measurement, :math:`\rho_\psi` is the state
preparation, :math:`\Lambda` is the average map over timesteps and gateset
elements, :math:`d` is the dimension of the system,  and where
:math:`F_\ave` is the average gate fidelity, taken over the gateset. The
functions :obj:`p` and :obj:`F` convert back and forth between depolarizing
parameters and fidelities.

An experiment parameter vector for this model is simply a specification of
:math:`m`, the length of the Clifford sequence used for that datum. Since
:class:`RandomizedBenchmarkingModel` represents 0/1 data, it is common to wrap
this model in a :class:`BinomialModel`:

>>> from qinfer import BinomialModel
>>> from qinfer import RandomizedBenchmarkingModel
>>> model = BinomialModel(RandomizedBenchmarkingModel(order=0, interleaved=False))
>>> expparams = np.array([
...    (100, 1000) # 1000 shots of sequences with length 100.
... ], dtype=model.expparams_dtype)

Interleaved Mode
~~~~~~~~~~~~~~~~

If one is interested in the fidelity of a single gate, rather than an entire
gateset, then the gate of interest can be interleaved with other gates from
the gateset to isolate its performance. In this mode, models admit an additional
model and experiment parameter, :math:`\tilde{p}` and ``mode``, respectively.
The :math:`\tilde{p}` model parameter is the depolarizing strength of the
twirl of the interleaved gate, such that the interleaved survival probability is
given by

.. math::

	\Pr(\text{survival} | \tilde{p}, p_{\text{ref}}, A_0, B_0; m, \text{interleaved}) = 
	A_0 (\tilde{p} p_{\text{ref}})^m + B_0.

Model instances for interleaved mode are constructed using the ``interleaved=True``
keyword argument:

>>> from qinfer.rb import RandomizedBenchmarkingModel
>>> model = RandomizedBenchmarkingModel(interleaved=True)

