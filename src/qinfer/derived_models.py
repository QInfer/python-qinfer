#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# derived_models.py: Models that decorate and extend other models.
##
# © 2017, Chris Ferrie (csferrie@gmail.com) and
#         Christopher Granade (cgranade@cgranade.com).
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the copyright holder nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
##

## FEATURES ###################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division # Ensures that a/b is always a float.

## ALL ########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
    'DerivedModel',
    'PoisonedModel',
    'BinomialModel',
    'GaussianHyperparameterizedModel',
    'MultinomialModel',
    'MLEModel',
    'RandomWalkModel',
    'GaussianRandomWalkModel'
]

## IMPORTS ####################################################################

from builtins import range
from functools import reduce
from past.builtins import basestring

import numpy as np
from scipy.stats import binom, multivariate_normal, norm
from itertools import combinations_with_replacement as tri_comb

from qinfer.utils import binomial_pdf, multinomial_pdf, sample_multinomial
from qinfer.abstract_model import Model, DifferentiableModel
from qinfer._lib import enum # <- TODO: replace with flufl.enum!
from qinfer.utils import binom_est_error
from qinfer.domains import IntegerDomain, MultinomialDomain

## FUNCTIONS ###################################################################

def rmfield( a, *fieldnames_to_remove ):
    # Removes named fields from a structured np array
    return a[ [ name for name in a.dtype.names if name not in fieldnames_to_remove ] ]


## CLASSES #####################################################################

class DerivedModel(Model):
    """
    Base class for any model that decorates another model.
    Provides passthroughs for modelparam_names, n_modelparams, etc.

    Many of these passthroughs can and should be overriden by
    specific subclasses, but it is rare that something will
    override all of them.
    """
    _underlying_model = None
    def __init__(self, underlying_model):
        self._underlying_model = underlying_model
        super(DerivedModel, self).__init__()
    
    @property
    def underlying_model(self):
        return self._underlying_model

    @property
    def base_model(self):
        return self._underlying_model.base_model

    @property
    def model_chain(self):
        return self._underlying_model.model_chain + (self._underlying_model, )

    @property
    def n_modelparams(self):
        # We have as many modelparameters as the underlying model.
        return self.underlying_model.n_modelparams

    @property
    def expparams_dtype(self):
        return self.underlying_model.expparams_dtype
        
    @property
    def modelparam_names(self):
        return self.underlying_model.modelparam_names

    @property
    def Q(self):
        return self.underlying_model.Q
    
    def clear_cache(self):
        self.underlying_model.clear_cache()

    def n_outcomes(self, expparams):
        return self.underlying_model.n_outcomes(expparams)
    
    def are_models_valid(self, modelparams):
        return self.underlying_model.are_models_valid(modelparams)

    def domain(self, expparams):
        return self.underlying_model.domain(expparams)
    
    def are_expparam_dtypes_consistent(self, expparams):
        return self.underlying_model.are_expparam_dtypes_consistent(expparams)
    
    def update_timestep(self, modelparams, expparams):
        return self.underlying_model.update_timestep(modelparams, expparams)

    def canonicalize(self, modelparams):
        return self.underlying_model.canonicalize(modelparams)

PoisonModes = enum.enum("ALE", "MLE")

class PoisonedModel(DerivedModel):
    r"""
    Model that simulates sampling error incurred by the MLE or ALE methods of
    reconstructing likelihoods from sample data. The true likelihood given by an
    underlying model is perturbed by a normally distributed random variable
    :math:`\epsilon`, and then truncated to the interval :math:`[0, 1]`.
    
    The variance of :math:`\epsilon` can be specified either as a constant,
    to simulate ALE (in which samples are collected until a given threshold is
    met), or as proportional to the variance of a possibly-hedged binomial
    estimator, to simulate MLE.
    
    :param Model underlying_model: The "true" model to be poisoned.
    :param float tol: For ALE, specifies the given error tolerance to simulate.
    :param int n_samples: For MLE, specifies the number of samples collected.
    :param float hedge: For MLE, specifies the hedging used in estimating the
        true likelihood.
    """
    def __init__(self, underlying_model,
        tol=None, n_samples=None, hedge=None
    ):
        super(PoisonedModel, self).__init__(underlying_model)
        
        if tol is None != n_samples is None:
            raise ValueError(
                "Exactly one of tol and n_samples must be specified"
            )
        
        if tol is not None:
            self._mode = PoisonModes.ALE
            self._tol = tol
        else:
            self._mode = PoisonModes.MLE
            self._n_samples = n_samples
            self._hedge = hedge if hedge is not None else 0.0
            
    ## METHODS ##

    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        
        # Get the original, undisturbed likelihoods.
        super(PoisonedModel, self).likelihood(outcomes, modelparams, expparams)
        L = self.underlying_model.likelihood(
            outcomes, modelparams, expparams)
            
        # Now get the random variates from a standard normal [N(0, 1)]
        # distribution; we'll rescale them soon.
        epsilon = np.random.normal(size=L.shape)
        
        # If ALE, rescale by a constant tolerance.
        if self._mode == PoisonModes.ALE:
            epsilon *= self._tol
        # Otherwise, rescale by the estimated error in the binomial estimator.
        elif self._mode == PoisonModes.MLE:
            epsilon *= binom_est_error(p=L, N=self._n_samples, hedge=self._hedge)
        
        # Now we truncate and return.
        np.clip(L + epsilon, 0, 1, out=L)
        return L
        
    def simulate_experiment(self, modelparams, expparams, repeat=1):
        """
        Simulates experimental data according to the original (unpoisoned)
        model. Note that this explicitly causes the simulated data and the
        likelihood function to disagree. This is, strictly speaking, a violation
        of the assumptions made about `~qinfer.abstract_model.Model` subclasses.
        This violation is by intention, and allows for testing the robustness
        of inference algorithms against errors in that assumption.
        """
        super(PoisonedModel, self).simulate_experiment(modelparams, expparams, repeat)
        return self.underlying_model.simulate_experiment(modelparams, expparams, repeat)

class BinomialModel(DerivedModel):
    """
    Model representing finite numbers of iid samples from another model,
    using the binomial distribution to calculate the new likelihood function.
    
    :param qinfer.abstract_model.Model underlying_model: An instance of a two-
        outcome model to be decorated by the binomial distribution.
        
    Note that a new experimental parameter field ``n_meas`` is added by this
    model. This parameter field represents how many times a measurement should
    be made at a given set of experimental parameters. To ensure the correct
    operation of this model, it is important that the decorated model does not
    also admit a field with the name ``n_meas``.
    """
    
    def __init__(self, underlying_model):
        super(BinomialModel, self).__init__(underlying_model)
        
        if not (underlying_model.is_n_outcomes_constant and underlying_model.n_outcomes(None) == 2):
            raise ValueError("Decorated model must be a two-outcome model.")
        
        if isinstance(underlying_model.expparams_dtype, str):
            # We default to calling the original experiment parameters "x".
            self._expparams_scalar = True
            self._expparams_dtype = [('x', underlying_model.expparams_dtype), ('n_meas', 'uint')]
        else:
            self._expparams_scalar = False
            self._expparams_dtype = underlying_model.expparams_dtype + [('n_meas', 'uint')]
    
    ## PROPERTIES ##
        
    @property
    def decorated_model(self):
        # Provided for backcompat only.
        return self.underlying_model
    

    @property
    def expparams_dtype(self):
        return self._expparams_dtype
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        return False
    
    ## METHODS ##
    
    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        return expparams['n_meas'] + 1

    def domain(self, expparams):
        """
        Returns a list of ``Domain``s, one for each input expparam.

        :param numpy.ndarray expparams:  Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property, or, in the case where ``n_outcomes_constant`` is ``True``,
            ``None`` should be a valid input.

        :rtype: list of ``Domain``
        """
        return [IntegerDomain(min=0,max=n_o-1) for n_o in self.n_outcomes(expparams)]
    
    def are_expparam_dtypes_consistent(self, expparams):
        """
        Returns `True` iff all of the given expparams 
        correspond to outcome domains with the same dtype.
        For efficiency, concrete subclasses should override this method 
        if the result is always `True`.

        :param np.ndarray expparams: Array of expparamms 
             of type `expparams_dtype`
        :rtype: `bool`
        """
        # The output type is always the same, even though the domain is not.
        return True

    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(BinomialModel, self).likelihood(outcomes, modelparams, expparams)
        pr1 = self.underlying_model.likelihood(
            np.array([1], dtype='uint'),
            modelparams,
            expparams['x'] if self._expparams_scalar else expparams)
        
        # Now we concatenate over outcomes.
        L = np.concatenate([
            binomial_pdf(expparams['n_meas'][np.newaxis, :], outcomes[idx], pr1)
            for idx in range(outcomes.shape[0])
            ]) 
        assert not np.any(np.isnan(L))
        return L
            
    def simulate_experiment(self, modelparams, expparams, repeat=1):
        # FIXME: uncommenting causes a slowdown, but we need to call
        #        to track sim counts.
        #super(BinomialModel, self).simulate_experiment(modelparams, expparams)
        
        # Start by getting the pr(1) for the underlying model.
        pr1 = self.underlying_model.likelihood(
            np.array([1], dtype='uint'),
            modelparams,
            expparams['x'] if self._expparams_scalar else expparams)
            
        dist = binom(
            expparams['n_meas'].astype('int'), # ← Really, NumPy?
            pr1[0, :, :]
        )
        sample = (
            (lambda: dist.rvs()[np.newaxis, :, :])
            if pr1.size != 1 else
            (lambda: np.array([[[dist.rvs()]]]))
        )
        os = np.concatenate([
            sample()
            for idx in range(repeat)
        ], axis=0)
        return os[0,0,0] if os.size == 1 else os
        
    def update_timestep(self, modelparams, expparams):
        return self.underlying_model.update_timestep(modelparams,
            expparams['x'] if self._expparams_scalar else expparams
        )
        
class DifferentiableBinomialModel(BinomialModel, DifferentiableModel):
    """
    Extends :class:`BinomialModel` to take advantage of differentiable
    two-outcome models.
    """
    
    def __init__(self, underlying_model):
        if not isinstance(underlying_model, DifferentiableModel):
            raise TypeError("Decorated model must also be differentiable.")
        BinomialModel.__init__(self, underlying_model)
    
    def score(self, outcomes, modelparams, expparams):
        raise NotImplementedError("Not yet implemented.")
        
    def fisher_information(self, modelparams, expparams):
        # Since the FI simply adds, we can multiply the single-shot
        # FI provided by the underlying model by the number of measurements
        # that we perform.
        two_outcome_fi = self.underlying_model.fisher_information(
            modelparams, expparams
        )
        return two_outcome_fi * expparams['n_meas']

class GaussianHyperparameterizedModel(DerivedModel):
    """
    Model representing a two-outcome model viewed through samples
    from one of two distinct Gaussian distributions.
    
    :param qinfer.abstract_model.Model underlying_model: An instance of a two-
        outcome model to be viewed through Gaussian distributions.
    """
    
    def __init__(self, underlying_model):
        super(GaussianHyperparameterizedModel, self).__init__(underlying_model)
        
        if not (underlying_model.is_n_outcomes_constant and underlying_model.n_outcomes(None) == 2):
            raise ValueError("Decorated model must be a two-outcome model.")
    
    ## PROPERTIES ##
        
    @property
    def decorated_model(self):
        # Provided for backcompat only.
        return self.underlying_model

    @property
    def modelparam_names(self):
        return self.underlying_model.modelparam_names + [
            r'\mu_0', r'\mu_1',
            r'\sigma_0^2', r'\sigma_1^2'
        ]
    
    @property
    def n_modelparams(self):
        return len(self.modelparam_names)
    
    ## METHODS ##
    
    def domain(self, expparams):
        """
        Returns a list of ``Domain``s, one for each input expparam.

        :param numpy.ndarray expparams:  Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property, or, in the case where ``n_outcomes_constant`` is ``True``,
            ``None`` should be a valid input.

        :rtype: list of ``Domain``
        """
        return [RealDomain()] * len(expparams)
    
    def are_expparam_dtypes_consistent(self, expparams):
        """
        Returns `True` iff all of the given expparams 
        correspond to outcome domains with the same dtype.
        For efficiency, concrete subclasses should override this method 
        if the result is always `True`.

        :param np.ndarray expparams: Array of expparamms 
             of type `expparams_dtype`
        :rtype: `bool`
        """
        return True

    def are_models_valid(self, modelparams):
        orig_mps = modelparams[:, :-4]
        sigma2 = modelparams[:, -2:]
        
        return np.all([
            self.underlying_model.are_models_valid(orig_mps),
            np.all(sigma2 > 0, axis=-1)
        ], axis=0)

    def underlying_likelihood(self, binary_outcomes, modelparams, expparams):
        """
        Given outcomes hypothesized for the underlying model, returns the likelihood
        which which those outcomes occur.
        """
        original_mps = modelparams[..., :-4]
        return self.underlying_model.likelihood(binary_outcomes, original_mps, expparams)

    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(GaussianHyperparameterizedModel, self).likelihood(outcomes, modelparams, expparams)
        pr0 = self.underlying_likelihood(
            np.array([0], dtype='uint'),
            modelparams,
            expparams
        )

        # We want these to broadcast to the shape
        #     (idx_underlying_outcome, idx_outcome, idx_modelparam, idx_experiment).
        # Thus, we need shape
        #     (idx_underlying_outcome,           1, idx_modelparam,              1).
        mu = (modelparams[:, -4:-2].T)[:, None, :, None]
        sigma = np.sqrt(
            (modelparams[:, -2:].T)[:, None, :, None]
        )

        assert np.all(sigma > 0)

        # Now we can rescale the outcomes to be random variates z drawn from N(0, 1).
        scaled_outcomes = (outcomes - mu) / sigma

        # We can then compute the conditional likelihood Pr(z | underlying_outcome, model).
        conditional_L = norm(0, 1).pdf(scaled_outcomes)

        # To find the marginalized likeihood, we now need the underlying likelihood
        # Pr(underlying_outcome | model), so that we can sum over the idx_u_o axis.
        # Note that we need to add a new axis to shift the underlying outcomes left
        # of the real-valued outcomes z.
        underlying_L = self.underlying_likelihood(
            np.array([0, 1], dtype='uint'),
            modelparams, expparams
        )[:, None, :, :]
        
        # Now we marginalize and return.
        L = (underlying_L * conditional_L).sum(axis=0)
        assert not np.any(np.isnan(L))
        return L
            
    def simulate_experiment(self, modelparams, expparams, repeat=1):
        super(GaussianHyperparameterizedModel, self).simulate_experiment(modelparams, expparams)
        
        # Start by generating a bunch of (0, 1) normalized random variates
        # that we'll randomly rescale to the right location and shape.
        zs = np.random.randn(modelparams.shape[0], expparams.shape[0])

        # Next, we sample a bunch of underlying outcomes to figure out
        # how to rescale everything.
        underlying_outcomes = self.underlying_model.simulate_experiment(
            modelparams[:, :-4], expparams
        )
        
        # We can now rescale zs to obtain the actual outcomes.
        mu = (modelparams[:, -4:-2].T)[:, None, :, None]
        sigma = np.sqrt(
            (modelparams[:, -2:].T)[:, None, :, None]
        )
        outcomes = (
            np.where(underlying_outcomes, mu[0], mu[1]) +
            np.where(underlying_outcomes, sigma[0], sigma[1]) * zs
        )

        return outcomes[0,0,0] if outcomes.size == 1 else outcomes

class MultinomialModel(DerivedModel):
    """
    Model representing finite numbers of iid samples from another model with 
    a fixed and finite number of outcomes,
    using the multinomial distribution to calculate the new likelihood function.
    
    :param qinfer.abstract_model.FiniteOutcomeModel underlying_model: An instance 
        of a D-outcome model to be decorated by the multinomial distribution. 
        This underlying model must have ``is_n_outcomes_constant`` as ``True``.
        
    Note that a new experimental parameter field ``n_meas`` is added by this
    model. This parameter field represents how many times a measurement should
    be made at a given set of experimental parameters. To ensure the correct
    operation of this model, it is important that the decorated model does not
    also admit a field with the name ``n_meas``.
    """
    
    ## INITIALIZER ##

    def __init__(self, underlying_model):
        super(MultinomialModel, self).__init__(underlying_model)

        if isinstance(underlying_model.expparams_dtype, str):
            # We default to calling the original experiment parameters "x".
            self._expparams_scalar = True
            self._expparams_dtype = [('x', underlying_model.expparams_dtype), ('n_meas', 'uint')]
        else:
            self._expparams_scalar = False
            self._expparams_dtype = underlying_model.expparams_dtype + [('n_meas', 'uint')]

        # Demand that the underlying model always has the same number of outcomes
        # This assumption could in principle be generalized, but not worth the effort now.
        assert(self.underlying_model.is_n_outcomes_constant)
        self._underlying_domain = self.underlying_model.domain(None)
        self._n_sides = self._underlying_domain.n_members
        # Useful for getting the right type, etc.
        self._example_domain = MultinomialDomain(n_elements=self.n_sides, n_meas=3) 

    ## PROPERTIES ##
    

    @property
    def decorated_model(self):
        # Provided for backcompat only.
        return self.underlying_model

    @property
    def expparams_dtype(self):
        return self._expparams_dtype  
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        # Different values of n_meas result in different numbers of outcomes
        return False

    @property
    def n_sides(self):
        """
        Returns the number of possible outcomes of the underlying model.
        """
        return self._n_sides

    @property
    def underlying_domain(self):
        """
        Returns the `Domain` of the underlying model.
        """
        return self._underlying_domain
    
    ## METHODS ##

    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        # Standard combinatorial formula equal to the number of 
        # possible tuples whose non-negative integer entries sum to n_meas.
        n = expparams['n_meas']
        k = self.n_sides
        return scipy.special.binom(n + k - 1, k - 1)

    def domain(self, expparams):
        """
        Returns a list of :class:`Domain` objects, one for each input expparam.
        :param numpy.ndarray expparams:  Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        :rtype: list of ``Domain``
        """
        return [
            MultinomialDomain(n_elements=self.n_sides, n_meas=ep['n_meas']) 
                for ep in expparams
        ]
    
    def are_expparam_dtypes_consistent(self, expparams):
        """
        Returns `True` iff all of the given expparams 
        correspond to outcome domains with the same dtype.
        For efficiency, concrete subclasses should override this method 
        if the result is always `True`.

        :param np.ndarray expparams: Array of expparamms 
             of type `expparams_dtype`
        :rtype: `bool`
        """
        # The output type is always the same, even though the domain is not.
        return True
  
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(MultinomialModel, self).likelihood(outcomes, modelparams, expparams)
        
        # Save a wee bit of time by only calculating the likelihoods of outcomes 0,...,d-2
        prs = self.underlying_model.likelihood(
            self.underlying_domain.values[:-1],
            modelparams,
            expparams['x'] if self._expparams_scalar else expparams) 
            # shape (sides-1, n_mps, n_eps)
        
        prs = np.tile(prs, (outcomes.shape[0],1,1,1)).transpose((1,0,2,3))
        # shape (n_outcomes, sides-1, n_mps, n_eps)

        os = self._example_domain.to_regular_array(outcomes)
        # shape (n_outcomes, sides)
        os = np.tile(os, (modelparams.shape[0],expparams.shape[0],1,1)).transpose((3,2,0,1))
        # shape (n_outcomes, sides, n_mps, n_eps)

        L = multinomial_pdf(os, prs) 
        assert not np.any(np.isnan(L))
        return L

    def simulate_experiment(self, modelparams, expparams, repeat=1):
        super(MultinomialModel, self).simulate_experiment(modelparams, expparams)
        
        n_sides = self.n_sides
        n_mps = modelparams.shape[0]
        n_eps = expparams.shape[0]

        # Save a wee bit of time by only calculating the likelihoods of outcomes 0,...,d-2
        prs = np.empty((n_sides,n_mps,n_eps))
        prs[:-1,...] = self.underlying_model.likelihood(
            self.underlying_domain.values[:-1],
            modelparams,
            expparams['x'] if self._expparams_scalar else expparams) 
            # shape (sides, n_mps, n_eps)


        os = np.concatenate([
            sample_multinomial(n_meas, prs[:,:,idx_n_meas], size=repeat)[np.newaxis,...]
            for idx_n_meas, n_meas in enumerate(expparams['n_meas'].astype('int'))
        ]).transpose((3,2,0,1))

        # convert to fancy data type
        os = self._example_domain.from_regular_array(os)

        return os[0,0,0] if os.size == 1 else os


class MLEModel(DerivedModel):
    r"""
    Uses the method of [JDD08]_ to approximate the maximum likelihood
    estimator as the mean of a fictional posterior formed by amplifying the
    Bayes update by a given power :math:`\gamma`. As :math:`\gamma \to
    \infty`, this approximation to the MLE improves, but at the cost of
    numerical stability.

    :param float likelihood_power: Power to which the likelihood calls
        should be rasied in order to amplify the Bayes update.
    """

    def __init__(self, underlying_model, likelihood_power):
        super(MLEModel, self).__init__(underlying_model)
        self._pow = likelihood_power

    def simulate_experiment(self, modelparams, expparams, repeat=1):
        super(MLEModel, self).simulate_experiment(modelparams, expparams, repeat)
        return self.underlying_model.simulate_experiment(modelparams, expparams, repeat)

    def likelihood(self, outcomes, modelparams, expparams):
        L = self.underlying_model.likelihood(outcomes, modelparams, expparams)
        return L**self._pow

class RandomWalkModel(DerivedModel):
    r"""
    Model such that after each time step, a random perturbation is added to
    each model parameter vector according to a given distribution.
    
    :param Model underlying_model: Model representing the likelihood with no
        random walk added.
    :param Distribution step_distribution: Distribution over step vectors.
    """
    def __init__(self, underlying_model, step_distribution):
        self._step_dist = step_distribution
        
        super(RandomWalkModel, self).__init__(underlying_model)
        
        if self.underlying_model.n_modelparams != self._step_dist.n_rvs:
            raise TypeError("Step distribution does not match model dimension.")
        
            
    ## METHODS ##
    
    def likelihood(self, outcomes, modelparams, expparams):
        super(RandomWalkModel, self).likelihood(outcomes, modelparams, expparams)
        return self.underlying_model.likelihood(outcomes, modelparams, expparams)
        
    def simulate_experiment(self, modelparams, expparams, repeat=1):
        super(RandomWalkModel, self).simulate_experiment(modelparams, expparams, repeat)
        return self.underlying_model.simulate_experiment(modelparams, expparams, repeat)
        
    def update_timestep(self, modelparams, expparams):
        # Note that the timestep update is presumed to be independent of the
        # experiment.
        steps = self._step_dist.sample(n=modelparams.shape[0] * expparams.shape[0])
        # Break apart the first two axes and transpose.
        steps = steps.reshape((modelparams.shape[0], expparams.shape[0], self.n_modelparams))
        steps = steps.transpose((0, 2, 1))
        
        return modelparams[:, :, np.newaxis] + steps
        
class GaussianRandomWalkModel(DerivedModel):
    r"""
    Model such that after each time step, a random perturbation is 
    added to each model parameter vector according to a 
    zero-mean gaussian distribution.
    
    The :math:`n\times n` covariance matrix of this distribution is 
    either fixed and known, or its entries are treated as unknown, 
    being appended to the model parameters.
    For diagonal covariance matrices, :math:`n` parameters are added to the model 
    storing the square roots of the diagonal entries of the covariance matrix.
    For dense covariance matrices, :math:`n(n+1)/2` parameters are added to 
    the model, storing the entries of the lower triangular portion of the
    Cholesky factorization of the covariance matrix.
    
    :param Model underlying_model: Model representing the likelihood with no
        random walk added.
    :param random_walk_idxs: A list or ``np.slice`` of 
        ``underlying_model`` model parameter indeces to add the random walk to.
        Indeces larger than ``underlying_model.n_modelparams`` should not 
        be touched.
    :param fixed_covariance: An ``np.ndarray`` specifying the fixed covariance 
        matrix (or diagonal thereof if ``diagonal`` is ``True``) of the 
        gaussian distribution. If set to ``None`` (default), this matrix is 
        presumed unknown and parameters are appended to the model describing 
        it.
    :param boolean diagonal: Whether the gaussian distribution covariance matrix
        is diagonal, or densely populated. Default is 
        ``True``.
    :param scale_mult: A function which takes an array of expparams and
        outputs a real number for each one, representing the scale of the 
        given experiment. This is useful if different experiments have 
        different time lengths and therefore incur different dispersion amounts.\
        If a string is given instead of a function, 
        thee scale multiplier is the ``exparam`` with that name.
    :param model_transformation: Either ``None`` or a pair of functions 
        ``(transform, inv_transform)`` specifying a transformation of ``modelparams``
        (of the underlying model) before gaussian noise is added, 
        and the inverse operation after
        the gaussian noise has been added.
    """
    def __init__(
            self, underlying_model, random_walk_idxs='all', 
            fixed_covariance=None, diagonal=True, 
            scale_mult=None, model_transformation=None
        ):
        
        self._diagonal = diagonal
        self._rw_idxs = np.s_[:underlying_model.n_modelparams] \
            if random_walk_idxs == 'all' else random_walk_idxs
            
        explicit_idxs = np.arange(underlying_model.n_modelparams)[self._rw_idxs]
        if explicit_idxs.size == 0:
            raise IndexError('At least one model parameter must take a random walk.')
    
        self._rw_names = [
                underlying_model.modelparam_names[idx] 
                for idx in explicit_idxs
            ]
        self._n_rw = len(explicit_idxs)
        
        self._srw_names = []
        if fixed_covariance is None:
            # In this case we need to lean the covariance parameters too,
            # therefore, we need to add modelparams
            self._has_fixed_covariance = False
            if self._diagonal:
                self._srw_names = [r"\sigma_{{{}}}".format(name) for name in self._rw_names]
                self._srw_idxs = (underlying_model.n_modelparams + \
                    np.arange(self._n_rw)).astype(np.int)
            else:
                self._srw_idxs = (underlying_model.n_modelparams +
                    np.arange(self._n_rw * (self._n_rw + 1) / 2)).astype(np.int)
                # the following list of indeces tells us how to populate 
                # a cholesky matrix with a 1D list of values
                self._srw_tri_idxs = np.tril_indices(self._n_rw)
                for idx1, name1 in enumerate(self._rw_names):
                    for name2 in self._rw_names[:idx1+1]:
                        if name1 == name2:
                            self._srw_names.append(r"\sigma_{{{}}}".format(name1))
                        else:
                            self._srw_names.append(r"\sigma_{{{},{}}}".format(name2,name1))
        else:
            # In this case the covariance matrix is fixed and fully specified
            self._has_fixed_covariance = True
            if self._diagonal:
                if fixed_covariance.ndim != 1:
                    raise ValueError('Diagonal covariance requested, but fixed_covariance has {} dimensions.'.format(fixed_covariance.ndim))
                if fixed_covariance.size != self._n_rw:
                    raise ValueError('fixed_covariance dimension, {}, inconsistent with number of parameters, {}'.format(fixed_covariance.size, self.n_rw))
                self._fixed_scale = np.sqrt(fixed_covariance)
            else:
                if fixed_covariance.ndim != 2:
                    raise ValueError('Dense covariance requested, but fixed_covariance has {} dimensions.'.format(fixed_covariance.ndim))
                if fixed_covariance.size != self._n_rw **2 or fixed_covariance.shape[-2] != fixed_covariance.shape[-1]:
                    raise ValueError('fixed_covariance expected to be square with width {}'.format(self._n_rw))
                self._fixed_chol = np.linalg.cholesky(fixed_covariance)
                self._fixed_distribution = multivariate_normal(
                    np.zeros(self._n_rw),
                    np.dot(self._fixed_chol, self._fixed_chol.T)
                )
                
        super(GaussianRandomWalkModel, self).__init__(underlying_model)
        
        if np.max(np.arange(self.n_modelparams)[self._rw_idxs]) > np.max(explicit_idxs):
            raise IndexError('random_walk_idxs out of bounds; must index (a subset of ) underlying_model modelparams.')
            
        if scale_mult is None:
            self._scale_mult_fcn = (lambda expparams: 1)
        elif isinstance(scale_mult, basestring):
            self._scale_mult_fcn = lambda x: x[scale_mult]
        else:
            self._scale_mult_fcn = scale_mult
            
        self._has_transformation = model_transformation is not None
        if self._has_transformation:
            self._transform = model_transformation[0]
            self._inv_transform = model_transformation[1]
            
        
                
    ## PROPERTIES ##
    
    @property
    def modelparam_names(self):
        return self.underlying_model.modelparam_names + self._srw_names
        
    @property 
    def n_modelparams(self):
        return len(self.modelparam_names)
        
    @property
    def is_n_outcomes_constant(self):
        return False
            
    ## METHODS ##
    
    def are_models_valid(self, modelparams):
        ud_valid = self.underlying_model.are_models_valid(modelparams[...,:self.underlying_model.n_modelparams])
        if self._has_fixed_covariance:
            return ud_valid
        elif self._diagonal:
            pos_std = np.greater_equal(modelparams[...,self._srw_idxs], 0).all(axis=-1)
            return np.logical_and(ud_valid, pos_std)
        else:
            return ud_valid
    
    def likelihood(self, outcomes, modelparams, expparams):
        super(GaussianRandomWalkModel, self).likelihood(outcomes, modelparams, expparams)
        return self.underlying_model.likelihood(outcomes, modelparams[...,:self.underlying_model.n_modelparams], expparams)
        
    def simulate_experiment(self, modelparams, expparams, repeat=1):
        super(GaussianRandomWalkModel, self).simulate_experiment(modelparams, expparams, repeat)
        return self.underlying_model.simulate_experiment(modelparams[...,:self.underlying_model.n_modelparams], expparams, repeat)
        
    def est_update_covariance(self, modelparams):
        """
        Returns the covariance of the gaussian noise process for one 
        unit step. In the case where the covariance is being learned,
        the expected covariance matrix is returned.
        
        :param modelparams: Shape `(n_models, n_modelparams)` shape array
        of model parameters.
        """
        if self._diagonal:
            cov = (self._fixed_scale ** 2 if self._has_fixed_covariance \
                else np.mean(modelparams[:, self._srw_idxs] ** 2, axis=0))
            cov = np.diag(cov)
        else:
            if self._has_fixed_covariance:
                cov = np.dot(self._fixed_chol, self._fixed_chol.T)
            else:
                chol = np.zeros((modelparams.shape[0], self._n_rw, self._n_rw))
                chol[(np.s_[:],) + self._srw_tri_idxs] = modelparams[:, self._srw_idxs]
                cov = np.mean(np.einsum('ijk,ilk->ijl', chol, chol), axis=0)
        return cov
        
    def update_timestep(self, modelparams, expparams):

        n_mps = modelparams.shape[0]
        n_eps = expparams.shape[0]
        if self._diagonal:
            scale = self._fixed_scale if self._has_fixed_covariance else modelparams[:, self._srw_idxs]
            # the following works when _fixed_scale has shape (n_rw) or (n_mps,n_rw)
            # in the latter, each particle gets dispersed by its own belief of the scale
            steps = scale * np.random.normal(size = (n_eps, n_mps, self._n_rw))
            steps = steps.transpose((1,2,0))
        else:
            if self._has_fixed_covariance:
                steps = np.dot(
                    self._fixed_chol, 
                    np.random.normal(size = (self._n_rw, n_mps * n_eps))
                ).reshape(self._n_rw, n_mps, n_eps).transpose((1,0,2))
            else:
                chol = np.zeros((n_mps, self._n_rw, self._n_rw))
                chol[(np.s_[:],) + self._srw_tri_idxs] = modelparams[:, self._srw_idxs]
                # each particle gets dispersed by its own belief of the cholesky
                steps = np.einsum('kij,kjl->kil', chol, np.random.normal(size = (n_mps, self._n_rw, n_eps)))
        
        # multiply by the scales of the current experiments
        steps = self._scale_mult_fcn(expparams) * steps
        
        if self._has_transformation:
            # repeat model params for every expparam
            new_mps = np.repeat(modelparams[np.newaxis,:,:], n_eps, axis=0).reshape((n_eps * n_mps, -1))
            # run transformation on underlying slice
            new_mps[:, :self.underlying_model.n_modelparams] = self._transform(
                    new_mps[:, :self.underlying_model.n_modelparams]
                )
            # add on the random steps to the relevant indeces
            new_mps[:, self._rw_idxs] += steps.transpose((2,0,1)).reshape((n_eps * n_mps, -1))
            #  back to regular parameterization
            new_mps[:, :self.underlying_model.n_modelparams] = self._inv_transform(
                    new_mps[:, :self.underlying_model.n_modelparams]
                )
            new_mps = new_mps.reshape((n_eps, n_mps, -1)).transpose((1,2,0))
        else:
            new_mps = np.repeat(modelparams[:,:,np.newaxis], n_eps, axis=2)
            new_mps[:, self._rw_idxs, :] += steps

        return new_mps

## TESTING CODE ###############################################################

if __name__ == "__main__":
    
    import operator as op
    from .test_models import SimplePrecessionModel
    
    m = BinomialModel(SimplePrecessionModel())
    
    os = np.array([6, 7, 8, 9, 10])
    mps = np.array([[0.1], [0.35], [0.77]])
    eps = np.array([(0.5 * np.pi, 10), (0.51 * np.pi, 10)], dtype=m.expparams_dtype)
    
    L = m.likelihood(
        os, mps, eps
    )
    print(L)
    
    assert m.call_count == reduce(op.mul, [os.shape[0], mps.shape[0], eps.shape[0]]), "Call count inaccurate."
    assert L.shape == (os.shape[0], mps.shape[0], eps.shape[0]), "Shape mismatch."
    
