#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# score.py: Provides mixins which compute the score numerically with a 
#   central difference.
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