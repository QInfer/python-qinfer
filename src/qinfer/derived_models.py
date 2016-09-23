#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# derived_models.py: Models that decorate and extend other models.
##
# © 2012 Chris Ferrie (csferrie@gmail.com) and
#        Christopher E. Granade (cgranade@gmail.com)
#     
# This file is a part of the Qinfer project.
# Licensed under the AGPL version 3.
##
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
    'MultinomialModel',
    'MLEModel',
    'RandomWalkModel'
]

## IMPORTS ####################################################################

from builtins import range
from functools import reduce

import numpy as np
from scipy.stats import binom

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
    
