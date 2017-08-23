#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# rb.py: Models for accelerated randomized benchmarking.
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

## ALL ########################################################################

__all__ = [
    'RandomizedBenchmarkingModel'
]

## IMPORTS ####################################################################

from itertools import starmap

import numpy as np
from qinfer._due import due, Doi
from qinfer.abstract_model import FiniteOutcomeModel, DifferentiableModel

from operator import mul

## FUNCTIONS ##################################################################

def p(F, d=2):
    """
    Given the fidelity of a gate in :math:`d` dimensions, returns the
    depolarizating probability of the twirled channel.
    
    :param float F: Fidelity of a gate.
    :param int d: Dimensionality of the Hilbert space on which the gate acts.
    """
    return (d * F - 1) / (d - 1)

def F(p, d=2):
    """
    Given the depolarizating probabilty of a twirled channel in :math:`d`
    dimensions, returns the fidelity of the original gate.
    
    :param float p: Depolarizing parameter for the twirled channel.
    :param int d: Dimensionality of the Hilbert space on which the gate acts.
    """
    return 1 - (1 - p) * (d - 1) / d

## CLASSES ####################################################################

class RandomizedBenchmarkingModel(FiniteOutcomeModel, DifferentiableModel):
    r"""
    Implements the randomized benchmarking or interleaved randomized
    benchmarking protocol, such that the depolarizing strength :math:`p`
    of the twirled channel is a parameter to be estimated, given a sequnce
    length :math:`m` as an experimental control. In addition, the zeroth-order
    "fitting"-parameters :math:`A` and :math:`B` are represented as model
    parameters to be estimated.
    
    :param bool interleaved: If `True`, the model implements the interleaved
        protocol, with :math:`\tilde{p}` being the depolarizing parameter for
        the interleaved gate and with :math:`p_{\text{ref}}` being the reference
        parameter.

    :modelparam p: Fidelity of the twirled error channel :math:`\Lambda`, represented as
        a decay rate :math:`p = (d F - 1) / (d - 1)`, where :math:`F`
        is the fidelity and :math:`d` is the dimension of the Hilbert space.
    :modelparam A: Scale of the randomized benchmarking decay, defined as
        :math:`\Tr[Q \Lambda(\rho - \ident / d)]`, where :math:`Q` is the final
        measurement, and where :math:`\ident` is the initial preparation.
    :modelparam B: Offset of the randomized benchmarking decay, defined as
        :math:`\Tr[Q \Lambda(\ident / d)]`.

    :expparam int m: Length of the randomized benchmarking sequence
        that was measured.
    """
    # TODO: add citations to the above docstring.

    @due.dcite(
        Doi("10.1088/1367-2630/17/1/013042"),
        description="Accelerated randomized benchmarking",
        tags=["implementation"]
    )
    def __init__(self, interleaved=False, order=0):
        self._il = interleaved
        if order != 0:
            raise NotImplementedError(
                "Only zeroth-order is currently implemented."
            )
        super(RandomizedBenchmarkingModel, self).__init__()

    @property
    def n_modelparams(self):
        return 3 + (1 if self._il else 0)
        
    @property
    def modelparam_names(self):
        return (
            # We want to know \tilde{p} := p_C / p, and so we make it
            # a model parameter directly. This means that later, we'll
            # need to extract p_C = p \tilde{p}.
            [r'\tilde{p}', 'p', 'A', 'B']
            if self._il else
            ['p', 'A', 'B']
        )
        
    @property
    def is_n_outcomes_constant(self):
        return True
    @property
    def expparams_dtype(self):
        return [('m', 'uint')] + (
            [('reference', bool)] if self._il else []
        )
    
    def n_outcomes(self, expparams):
        return 2
    
    def are_models_valid(self, modelparams):
        if self._il:
            p_C, p, A, B = modelparams.T
            return np.all([
                0 <= p,
                p <= 1,
                0 <= p_C,
                p_C <= 1,
                0 <= A,
                A <= 1,
                0 <= B,
                B <= 1,
                A + B <= 1,
                A * p + B <= 1,
                A * p_C + B <= 1
            ], axis=0)
        else:
            p, A, B = modelparams.T
            return np.all([
                0 <= p,
                p <= 1,
                0 <= A,
                A <= 1,
                0 <= B,
                B <= 1,
                A + B <= 1,
                A * p + B <= 1
            ], axis=0)
        
    def likelihood(self, outcomes, modelparams, expparams):
        super(RandomizedBenchmarkingModel, self).likelihood(outcomes, modelparams, expparams)
        
        if self._il:
            p_tilde, p, A, B = modelparams.T[:, :, np.newaxis]
            
            p_C = p_tilde * p
            
            p = np.where(expparams['reference'][np.newaxis, :], p, p_C)
        else:
            p, A, B = modelparams.T[:, :, np.newaxis]
            
        m = expparams['m'][np.newaxis, :]
        
        pr0 = np.zeros((modelparams.shape[0], expparams.shape[0]))
        pr0[:, :] = 1 - (A * (p ** m) + B)
        
        return FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)
        
    def score(self, outcomes, modelparams, expparams, return_L=False):

        na = np.newaxis
        n_m = modelparams.shape[0]
        n_e = expparams.shape[0]
        n_o = outcomes.shape[0]
        n_p = self.n_modelparams
        
        m = expparams['m'].reshape((1, 1, 1, n_e))
        
        L = self.likelihood(outcomes, modelparams, expparams)[na, ...]
        outcomes = outcomes.reshape((1, n_o, 1, 1))
        
        if not self._il:

            p, A, B = modelparams.T[:, :, np.newaxis]
            p = p.reshape((1, 1, n_m, 1))
            A = A.reshape((1, 1, n_m, 1))
            B = B.reshape((1, 1, n_m, 1))
        
            q = (-1)**(1-outcomes) * np.concatenate(np.broadcast_arrays(
                A * m * (p ** (m-1)), p**m, np.ones_like(p),
            ), axis=0) / L
            
        else:
        
            p_tilde, p_ref, A, B = modelparams.T[:, :, np.newaxis]
            p_C = p_tilde * p_ref
            
            mode = expparams['reference'][np.newaxis, :]
            
            p = np.where(mode, p_ref, p_C)
            
            p = p.reshape((1, 1, n_m, n_e))
            A = A.reshape((1, 1, n_m, 1))
            B = B.reshape((1, 1, n_m, 1))
        
            q = (-1)**(1-outcomes) * np.concatenate(np.broadcast_arrays(
                np.where(mode, 0, A * m * (p_tilde ** (m - 1)) * (p_ref ** m)),
                np.where(mode,
                    A * m * (p_ref ** (m - 1)),
                    A * m * (p_ref ** (m - 1)) * (p_tilde ** m)
                ),
                p**m, np.ones_like(p)
            ), axis=0) / L
        
        if return_L:
            # Need to strip off the extra axis we added for broadcasting to q.
            return q, L[0, ...]
        else:
            return q
