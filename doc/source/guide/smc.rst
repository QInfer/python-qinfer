..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _smc_guide:
    
.. currentmodule:: qinfer.smc

Sequential Monte Carlo
======================

Introduction
------------

Arguably the core of QInfer, the :mod:`qinfer.smc` module implements the
sequential Monte Carlo algorithm in a flexible and robust manner. At its most
basic, using QInfer's SMC implementation consists of specifying a model, a
prior, and a number of SMC particles to use.

The main component of QInfer's SMC support is the :class:`SMCUpdater` class,
which performs Bayesian updates on a given prior in response to new data.
In doing so, :class:`SMCUpdater` will also ensure that the posterior particles
are properly resampled. For more details on the SMC algorithm as implemented
by QInfer, please see [GFWC12]_.

Using :class:`SMCUpdater`
-------------------------

Creating and Configuring Updaters
"""""""""""""""""""""""""""""""""

The most straightfoward way of creating an :class:`SMCUpdater` instance is to
provide a model, a number of SMC particles and a prior distribution to choose
those particles from. Using the example of a :class:`~qinfer.test_models.SimplePrecessionModel`,
and a uniform prior :math:`\omega \sim \text{Uni}(0, 1)`:

>>> from qinfer.smc import SMCUpdater
>>> from qinfer.distributions import UniformDistribution
>>> from qinfer.test_models import SimplePrecessionModel
>>> model = SimplePrecessionModel()
>>> prior = UniformDistribution([0, 1])
>>> updater = SMCUpdater(model, 1000, prior)

Updating from Data
""""""""""""""""""

Once an updater has been created, one can then use it to update the prior
distribution to a posterior conditioned on experimental data. For example,

>>> true_model = prior.sample()
>>> experiment = np.array([12.1], dtype=model.expparams_dtype)
>>> outcome = model.simulate_experiment(true_model, experiment)
>>> updater.update(outcome, experiment)


Drawing Posterior Samples and Estimates
"""""""""""""""""""""""""""""""""""""""

Since :class:`SMCUpdater` inherits from :class:`~qinfer.distributions.Distribution`,
it can be sampled in the same way described in :ref:`distributions_guide`.

>>> posterior_samples = updater.sample(n=100)
>>> print(posterior_samples.shape)
(100, 1)

More commonly, however, one will want to calculate estimates such as
:math:`\hat{\vec{x}} = \mathbb{E}_{\vec{x}|\text{data}}[\vec{x}]`. These
estimates are given methods such as :meth:`~SMCUpdater.est_mean` and
:meth:`~SMCUpdater.est_covariance_mtx`.

>>> est = updater.est_mean()
>>> print(est)# doctest: +SKIP
[ 0.53147953]

Plotting Posterior Distributions
""""""""""""""""""""""""""""""""

TODO

Advanced Usage
--------------

Custom Resamplers
"""""""""""""""""

By default, :class:`SMCUpdater` uses the Liu and West resampling algorithm [LW01]_
with :math:`a = 0.98`. The resampling behavior can be controlled, however, by
passing different instances of :class:`~qinfer.resamplers.Resampler` to
:class:`SMCUpdater`. For instance, if one wants to create an updater with
:math:`a = 0.9` as was suggested by [WGFC13a]_:

>>> from qinfer.resamplers import LiuWestResampler
>>> updater = SMCUpdater(model, 1000, prior, resampler=LiuWestResampler(0.9))


Posterior Credible Regions
""""""""""""""""""""""""""

Posterior credible regions can be found by using the
:meth:`~SMCUpdater.est_credible_region` method. This method returns a set of
points :math:`\{\vec{x}_i'\}` such that the sum :math:`\sum_i w_i'` of the
corresponding weights :math:`\{w_i'\}` is at least a specified ratio of the
total weight.

This does not admit a very compact description, however, such that it is useful
to find region estimators :math:`\hat{X}` containing all of the particles
describing a credible region, as above. 

The :meth:`~SMCUpdater.region_est_hull` method does this by finding a convex
hull of the credible particles, while :meth:`~SMCUpdater.region_est_ellipsoid`
finds the minimum-volume enclosing ellipse (MVEE) of the convex hull region
estimator.

The derivation of these estimators, as well as a detailed discussion of their
performances, can be found in [GFWC12]_ and [Fer14]_.

Cluster Analysis
""""""""""""""""

TODO

Online Bayesian Cramer-Rao Bound Estimation
"""""""""""""""""""""""""""""""""""""""""""

TODO

Model Selection with Bayes Factors
""""""""""""""""""""""""""""""""""

When considering which of two models :math:`A` or :math:`B` best explains a
data record :math:`D`, the normalizations of SMC updates of the posterior
conditoned on each provide the probabilities :math:`\Pr(D | A)` and
:math:`\Pr(D | B)`. The normalization records can be obtained from the
:attr:`~SMCUpdater.normalization_record` properties of each. As the
probabilities of any individual data record quickly reach zero, however, it
becomes numerically unstable to consider these probabilities directly. Instead,
the property :attr:`~SMCUpdater.log_total_likelihood` records the quantity

.. math::
    
    \ell(D | M) = \sum_i \log \Pr(d_i | M)
    
for :math:`M \in \{A, B\}`. This is related to the Bayes factor
:math:`\text{f}` by

.. math::

    f = \exp(\ell(D | A) - \ell(D | B)).
    
As discussed in [WGFC13b]_, the Bayes factor tells which of the two models
under consideration is to be preferred as an explanation for the observed data.

