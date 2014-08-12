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

## FEATURES ##

from __future__ import division # Ensures that a/b is always a float.

## IMPORTS ##

import numpy as np

from utils import binomial_pdf

from abstract_model import Model, DifferentiableModel
    
## CLASSES #####################################################################

class SimplePrecessionModel(DifferentiableModel):
    r"""
    Describes the free evolution of a single qubit prepared in the
    :math:`\left|+\right\rangle` state under a Hamiltonian :math:`H = \omega \sigma_z / 2`,
    as explored in [GFWC12]_. (TODO: add other citations.)

    :param float min_freq: Minimum value for :math:`\omega` to accept as valid.
        This is used for testing techniques that mitigate the effects of degenerate models;
        there is no "good" reason to ever set this other than zero, other than to
        test with an explicitly broken model.
    """
    
    ## INITIALIZER ##

    def __init__(self, min_freq=0):
        super(SimplePrecessionModel, self).__init__()
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
        return 'float'
    
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
        super(SimplePrecessionModel, self).likelihood(outcomes, modelparams, expparams)
        
        # Possibly add a second axis to modelparams.
        if len(modelparams.shape) == 1:
            modelparams = modelparams[..., np.newaxis]
        
        # Allocating first serves to make sure that a shape mismatch later
        # will cause an error.
        pr0 = np.zeros((modelparams.shape[0], expparams.shape[0]))
        
        arg = np.dot(modelparams, expparams[..., np.newaxis].T) / 2        
        pr0 = np.cos(arg) ** 2
        
        # Now we concatenate over outcomes.
        return Model.pr0_to_likelihood_array(outcomes, pr0)

    def score(self, outcomes, modelparams, expparams):
        #TODO: vectorize this

        outcomes = outcomes[:, np.newaxis, np.newaxis]

        arg = modelparams * expparams / 2        
        return (
            ( expparams / np.tan(arg)) ** (outcomes) *
            (-expparams * np.tan(arg)) ** (1-outcomes)
        )
        
class NoisyCoinModel(Model):
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
    """
        
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
        return Model.pr0_to_likelihood_array(outcomes, pr0)
        
class NDieModel(Model):
    
    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return self.n
        
    @property
    def expparams_dtype(self):
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
    def __init__(self, n = 6):
	    self.n = n
	    Model.__init__(self)
	
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
        return self.n
    
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(NDieModel, self).likelihood(outcomes, modelparams, expparams)
        L = np.concatenate([np.array([modelparams[idx][outcomes]]) for idx in xrange(modelparams.shape[0])])
        return L[...,np.newaxis].transpose([1,0,2])
