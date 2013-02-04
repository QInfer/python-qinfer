#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# ale.py: Adaptive likelihood estimation utilities and models.
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

## FEATURES ####################################################################

from __future__ import division

## ALL #########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
    'ALEApproximateModel'
]

## IMPORTS #####################################################################

from itertools import count

import numpy as np
import warnings

from abstract_model import Model, Simulatable
from _exceptions import ApproximationWarning
from scipy.stats.distributions import binom

## FUNCTIONS ###################################################################

def binom_est_p(n, N, hedge=float(0)):
    return (n + hedge) / (N + 2 * hedge)

## CLASSES #####################################################################

class ALEApproximateModel(Model):
    # TODO: document
    
    def __init__(self, simulator,
        error_tol=1e-2, min_samp=10, samp_step=10,
        est_hedge=float(0), adapt_hedge=0.509
    ):
        # TODO: check that simulator always has two outcomes.
        self._simulator = simulator
        self._error_tol = error_tol
        self._min_samp = min_samp
        self._samp_step = samp_step
        # TODO: check that hedging is always non-negative.
        self._est_hedge = est_hedge
        self._adapt_hedge = adapt_hedge
        
    ## WRAPPED METHODS AND PROPERTIES ##
    # These methods and properties do nothing but pass along to the
    # consumed Simulatable instance, and so we present them here in a
    # compressed form.
    
    @property
    def n_modelparams(self): return self._simulator.n_modelparams
    @property
    def expparams_dtype(self): return self._simulator.expparams_dtype
    @property
    def is_n_outcomes_constant(self): return self._simulator.is_n_outcomes_constant
    @property
    def sim_count(self): return self._simulator.sim_count
    @property
    def Q(self): return self._simulator.Q
    
    def n_outcomes(self, expparams): return self._simulator.n_outcomes(expparams)
    def are_models_valid(self, modelparams): return self._simulator.are_models_valid(modelparams)
    def simulate_experiment(self, modelparams, expparams, repeat=1):
        return self._simulator.simulate_experiment(modelparams, expparams, repeat)
    def experiment_cost(self, expparams): return self._simulator.experiment_cost(expparams)
    
    ## IMPLEMENTATIONS OF MODEL METHODS ##
    
    def likelihood(self, outcomes, modelparams, expparams):
        # FIXME: at present, will proceed until ALL model experiment pairs
        #        are below error tol.
        #        Should disable one-by-one, but that's tricky.
        super(ALEApproximateModel, self).likelihood(outcomes, modelparams, expparams)
        # We will use the fact we have assumed a two-outcome model to make the
        # problem easier. As such, we will rely on the static method 
        # Model.pr0_to_likelihood_array.
        
        # Start off with min_samp samples.
        n = np.zeros((modelparams.shape[0], expparams.shape[0]))
        for N in count(start=self._min_samp, step=self._samp_step):
            sim_data = self._simulator.simulate_experiment(
                modelparams, expparams, repeat=self._samp_step
            )
            n += np.sum(sim_data, axis=0) # Sum over the outcomes axis to find the
                                          # number of 1s.
            error_est_p1 = binom_est_p(n, N, self._adapt_hedge)
            if np.all(error_est_p1 < self._error_tol): break
            
        return Model.pr0_to_likelihood_array(outcomes, 1 - binom_est_p(n, N, self._est_hedge))
    
