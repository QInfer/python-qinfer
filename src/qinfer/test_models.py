#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_models.py: Simple models for testing inference engines.
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

## FEATURES ##################################################################

from __future__ import absolute_import
from __future__ import division # Ensures that a/b is always a float.

## EXPORTS ###################################################################

__all__ = [
    'SimpleInversionModel',
    'SimplePrecessionModel',
    'CoinModel',
    'NoisyCoinModel',
    'NDieModel'
]

## IMPORTS ###################################################################

from builtins import range

import numpy as np

from .utils import binomial_pdf

from .abstract_model import FiniteOutcomeModel, DifferentiableModel
    
## CLASSES ####################################################################

class SimpleInversionModel(FiniteOutcomeModel, DifferentiableModel):
    r"""
    Describes the free evolution of a single qubit prepared in the
    :math:`\left|+\right\rangle` state under a Hamiltonian :math:`H = \omega \sigma_z / 2`,
    using the interactive QLE model proposed by [WGFC13a]_.

    :param float min_freq: Minimum value for :math:`\omega` to accept as valid.
        This is used for testing techniques that mitigate the effects of
        degenerate models; there is no "good" reason to ever set this other
        than zero, other than to test with an explicitly broken model.
    """
    
    ## INITIALIZER ##

    def __init__(self, min_freq=0):
        super(SimpleInversionModel, self).__init__()
        self._min_freq = min_freq

    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return 1
    
    @property
    def modelparam_names(self):
        return [r'\omega']
        
    @property
    def expparams_dtype(self):
        return [('t', 'float'), ('w_', 'float')]
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        return True
    
    ## METHODS ##
    
    def are_models_valid(self, modelparams):
        return np.all(modelparams > self._min_freq, axis=1)
    
    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        return 2
    
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(SimpleInversionModel, self).likelihood(
            outcomes, modelparams, expparams
        )

        # Possibly add a second axis to modelparams.
        if len(modelparams.shape) == 1:
            modelparams = modelparams[..., np.newaxis]
            
        t = expparams['t']
        dw = modelparams - expparams['w_']
        
        # Allocating first serves to make sure that a shape mismatch later
        # will cause an error.
        pr0 = np.zeros((modelparams.shape[0], expparams.shape[0]))
        pr0[:, :] = np.cos(t * dw / 2) ** 2
        
        # Now we concatenate over outcomes.
        return FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)

    def score(self, outcomes, modelparams, expparams, return_L=False):
        if len(modelparams.shape) == 1:
            modelparams = modelparams[:, np.newaxis]
            
        t = expparams['t']
        dw = modelparams - expparams['w_']

        outcomes = outcomes.reshape((outcomes.shape[0], 1, 1))

        arg = dw * t / 2        
        q = (
            np.power( t / np.tan(arg), outcomes) *
            np.power(-t * np.tan(arg), 1 - outcomes)
        )[np.newaxis, ...]

        assert q.ndim == 4
        
        
        if return_L:
            return q, self.likelihood(outcomes, modelparams, expparams)
        else:
            return q


class SimplePrecessionModel(SimpleInversionModel):
    r"""
    Describes the free evolution of a single qubit prepared in the
    :math:`\left|+\right\rangle` state under a Hamiltonian :math:`H = \omega \sigma_z / 2`,
    as explored in [GFWC12]_.

    :param float min_freq: Minimum value for :math:`\omega` to accept as valid.
        This is used for testing techniques that mitigate the effects of
        degenerate models; there is no "good" reason to ever set this to be
        less than zero, other than to test with an explicitly broken model.

    :modelparam omega: The precession frequency :math:`\omega`.
    :scalar-expparam float: The evolution time :math:`t`.
    """
        
    @property
    def expparams_dtype(self):
        return 'float'
    
    def likelihood(self, outcomes, modelparams, expparams):
        # Pass the expparams to the superclass as a record array.
        new_eps = np.empty(expparams.shape, dtype=super(SimplePrecessionModel, self).expparams_dtype)
        new_eps['w_'] = 0
        new_eps['t'] = expparams

        return super(SimplePrecessionModel, self).likelihood(outcomes, modelparams, new_eps)

    def score(self, outcomes, modelparams, expparams, return_L=False):
        # Pass the expparams to the superclass as a record array.
        new_eps = np.empty(expparams.shape, dtype=super(SimplePrecessionModel, self).expparams_dtype)
        new_eps['w_'] = 0
        new_eps['t'] = expparams

        return super(SimplePrecessionModel, self).score(outcomes, modelparams, new_eps, return_L)
           
class CoinModel(FiniteOutcomeModel, DifferentiableModel):
    r"""
    Arguably the simplest possible model; the unknown model parameter 
    is the bias of a coin, and an experiment consists of flipping it and 
    looking at the result.
    The model parameter :math:`p` represents the probability of outcome 0.
    """

    ## INITIALIZER ##

    def __init__(self):
        super(CoinModel, self).__init__()

    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return 1
    
    @property
    def modelparam_names(self):
        return [r'p']
        
    @property
    def expparams_dtype(self):
        return []
    
    @property
    def is_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a FiniteOutcomeModel instance.
        """
        return True
    
    ## METHODS ##
    
    @staticmethod
    def are_models_valid(modelparams):
        return np.logical_and(modelparams >= 0, modelparams <= 1).all(axis=1)
 
    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        return 2
    
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(CoinModel, self).likelihood(outcomes, modelparams, expparams)
                  
        # Our job is easy.
        pr0 = np.tile(modelparams.flatten(), (expparams.shape[0], 1)).T
        
        # Now we concatenate over outcomes.
        return FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)

    def score(self, outcomes, modelparams, expparams, return_L=False):

        p = modelparams.flatten()[np.newaxis, :]
        side = outcomes.flatten()[:, np.newaxis]

        q = (1 - side) / p - side / (1 - p)

        #  we need to add singleton dimension since there 
        # is only one model param
        q = q[np.newaxis, :, :]

        # duplicate this for any exparams we have
        q = np.tile(q, (expparams.shape[0], 1, 1, 1)).transpose((1,2,3,0))
        
        if return_L:
            return q, self.likelihood(outcomes, modelparams, expparams)
        else:
            return q

class NoisyCoinModel(FiniteOutcomeModel):
    r"""
    Implements the "noisy coin" model of [FB12]_, where the model parameter
    :math:`p` is the probability of the noisy coin. This model has two
    experiment parameters, :math:`\alpha` and :math:`\beta`, which are the
    probabilities of observing a "0" outcome conditoned on the "true" outcome
    being 0 and 1, respectively. That is, for an ideal coin, :math:`\alpha = 1`
    and :math:`\beta = 0`.
    
    Note that :math:`\alpha` and :math:`\beta` are implemented as experiment
    parameters not because we expect to design over those values, but because
    a specification of each is necessary to honestly describe an experiment
    that was performed.

    :modelparam p: "Heads" probability :math:`p`.
    :expparam float alpha: Visibility parameter :math:`\alpha`.
    :expparam float beta: Visibility parameter :math:`\beta`.
    """
        
    def __init__(self):
        super(NoisyCoinModel, self).__init__()

    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return 1
        
    @property
    def expparams_dtype(self):
        return [('alpha','float'), ('beta','float')]
    
    @property
    def is_n_outcomes_constant(self):
        return True
    
    ## METHODS ##
    
    @staticmethod
    def are_models_valid(modelparams):
        return np.logical_and(modelparams >= 0, modelparams <= 1).all(axis=1)
    
    def n_outcomes(self, expparams):
        return 2
    
    def likelihood(self, outcomes, modelparams, expparams):
        # Unpack alpha and beta.
        a = expparams['alpha']
        b = expparams['beta']
        
        # Find the probability of getting a "0" outcome.
        pr0 = modelparams * a + (1 - modelparams) * b
        
        # Concatenate over outcomes.
        return FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)
        
class NDieModel(FiniteOutcomeModel):
    r"""
    Implements a model of rolling a die with n sides,
    whose unknown model parameters are the weights 
    of each side; a generalization of CoinModel. An 
    experiment consists of rolling the die once. The 
    faces of the die are zero indexed, labeled 0,1,2,...,n-1.

    :param int n: Number of sides on the die.
    :param float threshold: How close to 1 the probabilites of the sides of the die must be.
    """

    ## INITIALIZERS ##

    def __init__(self, n=6, threshold=1e-7):
        # We need to set this private property before
        # calling super, which relies on n_modelparams
        self._n = n
        super(NDieModel, self).__init__()
        self._threshold = threshold

    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return self._n
        
    @property
    def expparams_dtype(self):
        # This is a dummy parameter, its value doesn't come 
        # into the likelihood.
        return [('exp_num', 'int')]
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        return True
    
    ## METHODS ##
    
    def are_models_valid(self, modelparams):
        sums = np.abs(np.sum(modelparams, axis=1) - 1) <= self._threshold
        bounds = np.logical_and(modelparams >= 0, modelparams <= 1).all(axis=1)
        return np.logical_and(sums, bounds)
    
    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        return self._n 
    
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(NDieModel, self).likelihood(outcomes, modelparams, expparams)
        # Like for CoinModel, the modelparams _are_ the likelihoods;
        # we just need to do some tedious reshaping and tiling.
        L = np.concatenate([np.array([modelparams[idx][outcomes]]) for idx in range(modelparams.shape[0])])
        return np.tile(L[np.newaxis,...],(expparams.shape[0],1,1)).transpose((2,1,0))
