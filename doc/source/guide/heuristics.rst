..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _heuristics_guide:
    
.. currentmodule:: qinfer

Experiment Design Heuristics
============================

Using Heuristics in Updater Loops
---------------------------------

During an experiment, the current posterior distribution represented by
an :class:`~qinfer.SMCUpdater` instance can be used to *adaptively*
make decisions about which measurements should be performed. For example,
utility functions such as the information gain and negative variance
can be used to choose optimal measurements.

On the other hand, this optimization is generally quite computationally
expensive, such that many less optimal measurements could have been performed
before an optimization step completes. Philosophically, there are no "bad"
experiments, in that even suboptimal measurements can still carry useful
information about the parameters of interest.

In light of this, it can be
substantially less expensive to use a *heuristic* function of prior information
to select experiments without explicit optimization. For example, consider
the single-parameter inversion precession model with model parameters
:math:`\vec{x} = (\omega)`, experiment parameters :math:`e = (\omega_-, t)`
and likelihood function

.. math::

    \Pr(1 | \omega; \omega_-, t) = \sin^2([\omega - \omega_-] t / 2).

For a given posterior distribution,
the *particle guess heuristic* (PGH) [WGFC13a]_ then chooses
:math:`\omega_-` and :math:`t` for the next experiment by
first sampling two particles :math:`\omega_-` and :math:`\omega_-'` from the posterior.
The PGH then assigns the time :math:`t = 1 / |\omega_- \omega_-'|`.

**QInfer** implements heuristics as subtypes of :class:`~qinfer.Heuristic`,
each of which take an updater and can be called to produce experimental
parameters. For example, the PGH is implemented by the class :class:`~qinfer.PGH`,
and can be used in an updater loop to adaptively select experiments.

.. plot::

    model = SimpleInversionModel()
    prior = UniformDistribution([0, 1])
    updater = SMCUpdater(model, 1000, prior)
    heuristic = PGH(updater, inv_field='w_', t_field='t')

    true_omega = prior.sample()

    ts = []
    est_omegas = []

    for idx_exp in range(100):
        experiment = heuristic()
        datum = model.simulate_experiment(true_omega, experiment)
        updater.update(datum, experiment)

        ts.append(experiment['t'])
        est_omegas.append(updater.est_mean())

    ax = plt.subplot(2, 1, 1)
    plt.semilogy((est_omegas - true_omega) ** 2)
    plt.ylabel('Squared Error')

    plt.subplot(2, 1, 2, sharex=ax)
    plt.semilogy(ts)
    plt.xlabel('# of Measurements')
    plt.ylabel('$t$')

    plt.show() 

Changing Heuristic Parameters
-----------------------------

Essentially, heuristics in **QInfer** are functions that take :class:`SMCUpdater`
instances and return functions that then yield experiments. This design allows
for specializing heuristics by providing other arguments along with the
updater. For instance, the :class:`ExpSparseHeuristic` class implements
exponentially-sparse sampling :math:`t_k = ab^k` for :class:`SimplePrecessionModel`.
Both :math:`a` and :math:`b` are parameters of the heuristic, named
``scale`` and ``base``, respectively. Thus, it is easy to override the
defaults to obtain different heuristics.

.. plot::

    model = SimplePrecessionModel()
    prior = UniformDistribution([0, 1])
    updater = SMCUpdater(model, 1000, prior)
    heuristic = ExpSparseHeuristic(updater, scale=0.5)

    true_omega = prior.sample()
    est_omegas = []

    for idx_exp in range(100):
        experiment = heuristic()
        datum = model.simulate_experiment(true_omega, experiment)
        updater.update(datum, experiment)

        est_omegas.append(updater.est_mean())

    plt.semilogy((est_omegas - true_omega) ** 2)
    plt.xlabel('# of Measurements')
    plt.ylabel('Squared Error')

    plt.show() 

In overriding the default parameters of heuristics, the
:func:`functools.partial` function provided with the Python
standard library is especially useful, as it allows for easily
making new heuristics with different default parameter values:

>>> from qinfer import ExpSparseHeuristic
>>> from functools import partial
>>> rescaled_heuristic_class = partial(ExpSparseHeuristic, scale=0.01)

Later, once we have an updater,
we can then make new instances of our rescaled heuristic.

>>> heuristic = rescaled_heuristic_class(updater) # doctest: +SKIP

This technique is especially useful in :ref:`perf_testing`, as
it makes it easy to modify existing heuristics by changing default
parameters through partial application. 
