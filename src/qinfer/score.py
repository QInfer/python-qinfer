#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# score.py: Provides mixins which compute the score numerically with a 
#   central difference.
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
from __future__ import division

## IMPORTS ####################################################################

from builtins import range

import numpy as np
    
## CLASSES ####################################################################

class ScoreMixin(object):
    r"""
    A mixin which includes a method ``score`` that numerically estimates the
    score of the likelihood function. Any class which mixes in this class 
    should be equipped with a property ``n_modelparams`` and a method 
    ``likelihood`` to satisfy dependency.
    """
    
    
    _h = 1e-10
    
    @property
    def h(self):
        r"""
        Returns the step size to be used in numerical differentiation with 
        respect to the model parameters.
        
        The step size is given as a vector with length ``n_modelparams`` so 
        that each model parameter can be weighted independently.
        """
        if np.size(self._h) > 1:
            assert np.size(self._h) == self.n_modelparams
            return self._h
        else:
            return self._h * np.ones(self.n_modelparams)
    
    def score(self, outcomes, modelparams, expparams, return_L=False):
        r"""
        Returns the numerically computed score of the likelihood 
        function, defined as:
        
        .. math::
        
            q(d, \vec{x}; \vec{e}) = \vec{\nabla}_{\vec{x}} \log \Pr(d | \vec{x}; \vec{e}).
            
        Calls are represented as a four-index tensor
        ``score[idx_modelparam, idx_outcome, idx_model, idx_experiment]``.
        The left-most index may be suppressed for single-parameter models.
        
        The numerical gradient is computed using the central difference method, 
        with step size given by the property `~ScoreMixin.h`.
        
        If return_L is True, both `q` and the likelihood `L` are returned as `q, L`.
        """
        
        if len(modelparams.shape) == 1:
            modelparams = modelparams[:, np.newaxis]
        
        # compute likelihood at central point
        L0 = self.likelihood(outcomes, modelparams, expparams)
        
        # allocate space for the score
        q = np.empty([self.n_modelparams, 
                      outcomes.shape[0], 
                      modelparams.shape[0], 
                      expparams.shape[0]])
        h_perturb = np.empty(modelparams.shape)
        
        # just loop over the model parameter as there usually won't be so many
        # of them that vectorizing would be worth the effort.
        for mp_idx in range(self.n_modelparams):
            h_perturb[:] = np.zeros(modelparams.shape)
            h_perturb[:, mp_idx] = self.h[mp_idx]
            # use the chain rule since taking the numerical derivative of a 
            # logarithm is unstable
            q[mp_idx, :] = (
                self.likelihood(outcomes, modelparams + h_perturb, expparams) - 
                self.likelihood(outcomes, modelparams - h_perturb, expparams)
            ) / (2 * self.h[mp_idx] * L0)
            
        
        if return_L:
            return q, L0
        else: 
            return q