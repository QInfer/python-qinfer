..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _tomography_guide:
    
.. currentmodule:: qinfer.tomography

Quantum Tomography
==================

Introduction
------------

Tomography is the most common quantum statistical problem being the subject
of both theoretical and practical studies. The :mod:`qinfer.tomography` module
has rich support for many of the common models of tomography including
standard distributions and heuristics, and also provides convenient plotting
tools.

If the quantum state of a system is :math:`\rho` and a measurement is obtained, then the probability of obtaining the outcome associated with effect :math:`E` is :math:`\operatorname{Pr}(E|\rho) = \operatorname{Tr}(\rho E)`, the Born rule. The tomography problem is the inverse problem and is succinctly stated as follows. Given an unknown state :math:`\rho`, and a set of count statistics :math:`\{n_k\}` from measurements, the corresponding operators of which are :math:`\{E_k\}`, determine :math:`\rho`. 

In the context of Bayesian statistics, priors are distributions of quantum states and models define the Hilbert space dimension and the Born rule as a likelihood function (and perhaps additional complications associated with drift parameters and measurement errors).
The tomography module implements these details and has built-in support for common specifications.

The tomography module was developed concurrently with new results on Bayesian
priors in [GCC16]_. Please see the paper for more detailed discussions of these
more advanced topics.

QuTiP
~~~~~

Note that the Tomography module requires `QuTiP <http://qutip.org/>`_ which must be installed separately. Rather than reimplementing common operations on quantum states, we make use of QuTiP `Quantum objects <http://qutip.org/docs/3.1.0/guide/guide-basics.html>`_. For many simple use cases the QuTiP dependency is not exposed, but familiarity with `Quantum objects <http://qutip.org/docs/3.1.0/guide/guide-basics.html>`_ would be required to implement new distributions, models or heuristics.

Bases
-----

Bases define the map from abstract quantum theory to concrete representations. Once a basis is chosen, the tomography problem becomes a special case of a generic parameter estimation problem.

The tomography module is used in the same way as other but requires the specification of a basis via the :class:`TomographyBasis` class. QInfer comes with the following :ref:`tomography_bases`: the `Gell-Mann basis <https://en.wikipedia.org/wiki/Gell-Mann_matrices>`_, the `Pauli basis <https://en.wikipedia.org/wiki/Pauli_matrices>`_ and function to combine bases with the tensor product. Thus, the first step in using the tomography module is to define a basis:

>>> from qinfer.tomography import pauli_basis
>>> basis = pauli_basis(1)
>>> print(basis) #doctest: +ELLIPSIS
<TomographyBasis pauli_basis dims=[2] at ...>

User defined bases must be orthogonal Hermitian operators and have a 0'th component of :math:`I/\sqrt{d}`, where :math:`d` is the dimension of the quantum system and :math:`I` is the identity operator. This implies the remaining operators are traceless.

Built-in Distributions
----------------------

QInfer comes with several built-in distributions listed in :ref:`tomography_distributions`. Each of these is a subclass of :class:`DensityOperatorDistribution`. Distributions of quantum channels can also be subclassed as such with appeal to the Choi-Jamiolkowski isomorphism.

For density matrices, the :class:`GinibreDistribution` defines a prior over mixed quantum which allows for support also for states of fixed rank [OSZ10]_.

>>> from qinfer.tomography import pauli_basis, GinibreDistribution
>>> basis = pauli_basis(1)
>>> prior = GinibretDistribution(basis)
>>> print(prior.sample()) #doctest: +SKIP
[[ 0.70710678 -0.17816233  0.45195168 -0.08341437]]


Plotting tools
~~~~~~~~~~~~~~




Using :class:`TomographyModel`
------------------------------



Built-in Heuristics
-------------------
