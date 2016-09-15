#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# expdesign.py: Experiment design heuristics specialized for state and
#     process tomography.
##
# © 2015 Chris Ferrie (csferrie@gmail.com) and
#        Christopher E. Granade (cgranade@cgranade.com).
# Based on work with Joshua Combes (joshua.combes@gmail.com).
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

# TODO: docstrings!
# TODO: unit tests!

## DOCSTRING #################################################################

"""
"""

## FEATURES ##################################################################

from __future__ import absolute_import
from __future__ import division

## IMPORTS ###################################################################

from builtins import range

from qinfer import Heuristic
from qinfer.tomography.bases import pauli_basis

import numpy as np

from abc import abstractmethod

# Since the rest of QInfer does not require QuTiP,
# we need to import it in a way that we don't propagate exceptions if QuTiP
# is missing or is too early a version.
from qinfer.utils import get_qutip_module
qt = get_qutip_module('3.2')

## EXPORTS ###################################################################
# TODO

## FUNCTIONS #################################################################

# TODO: document, contribute to QuTiP?
def heisenberg_weyl_operators(d=2):
    w = np.exp(2 * np.pi * 1j / d)
    X = qt.Qobj([
        qt.basis(d, (idx + 1) % d).data.todense().view(np.ndarray)[:, 0] for idx in range(d)
    ])
    Z = qt.Qobj(np.diag(w ** np.arange(d)))
    
    return [X**i * Z**j for i in range(d) for j in range(d)]

## CLASSES ####################################################################

class StateTomographyHeuristic(Heuristic):
    # TODO: docstring
    # for when I write that docstring, basis is provided so that this can
    # be used to define measurement parts of the process tomography experiments.
    def __init__(self, updater, basis=None, other_fields=None):     
        self._up = updater
        self._other_fields = {} if other_fields is None else other_fields
        self._dim = updater.model.base_model.dim
        self._basis = updater.model.base_model.basis if basis is None else basis
        
    def __call__(self):
        expparams = np.zeros((1,), dtype=self._up.model.expparams_dtype)
        expparams['meas'][0, :] = self._basis.state_to_modelparams(
            self._next_measurement()
        )

        for field, value in self._other_fields.items():
                expparams[field] = value
        
        return expparams
    
    @abstractmethod
    def _next_measurement(self):
        pass
    

class RandomStabilizerStateHeuristic(StateTomographyHeuristic):
    """
    Randomly chooses rank-1 projectors onto a stabilizer state.
    """
    
    # This heuristic isn't terribly *efficient*, but the heuristic
    # gets called far less than the likelihood itself, so that shouldn't
    # dominate.
    
    def __init__(self, updater, basis=None, other_fields=None):
        super(RandomStabilizerStateHeuristic, self).__init__(
            updater, basis, other_fields
        )
    
        self._hw_group = heisenberg_weyl_operators(self._basis.dim)
    
    def _next_measurement(self):
        return qt.ket2dm(
            np.random.choice(np.random.choice(
                self._hw_group
            ).eigenstates()[1])
        )


class RandomPauliHeuristic(StateTomographyHeuristic):
    """
    Randomly chooses a Pauli measurement. Defined for qubits only.
    """
    # This heuristic is likewise a bit silly, as it draws a
    # random Pauli then decomposes it into what is almost
    # certainly a Pauli basis. We do this, however, in the interest
    # of generality. Someone could have used a different basis
    # to define their tomography model, after all.
    def __init__(self, updater, basis=None, other_fields=None):
        super(RandomPauliHeuristic, self).__init__(
            updater, basis, other_fields
        )
        
        nq = int(np.log2(self._dim))
        if 2**nq != self._dim:
            raise ValueError("RandomPauliHeuristic is defined only for qubits.")
        
        self._nq = nq
        self._pauli_basis = pauli_basis(nq)
        
    def _next_measurement(self):
        # Remember, the basis elements are normalized to 1 / sqrt(d).
        return (
            self._pauli_basis[0] +
            self._pauli_basis[1 + np.random.choice(len(self._pauli_basis) - 1)]
        ) * np.sqrt(self._dim) / 2


class ProcessTomographyHeuristic(Heuristic):
    def __init__(self, updater, basis, other_fields=None):     
        self._up = updater
        self._other_fields = {} if other_fields is None else other_fields
        self._dim = updater.model.base_model.dim
        self._basis = basis
        self._channel_basis = updater.model.base_model.basis
        
    def __call__(self):
        expparams = np.zeros((1,), dtype=self._up.model.expparams_dtype)
        expparams['meas'][0, :] = self._channel_basis.state_to_modelparams(
            # By PBT (Apx A), we need to multiply the state by a factor of
            # D for this to work. Here, however, dim means the channel,
            # so we need a square root.
            np.sqrt(self._dim) * qt.tensor(*self._next_prepmeas())
        )

        for field, value in self._other_fields.items():
                expparams[field] = value
        
        return expparams
    
    @abstractmethod
    def _next_prepmeas(self):
        """
        Implementing subclasses should return a tuple ``(preparation, measurement)``
        of two Qobjs, normalized in the 1- and ∞-norms, respectively.
        """
        pass
    
class ProductHeuristic(ProcessTomographyHeuristic):
    """
    Takes two heuristic classes, one for preparations
    and one for measurements, then returns a sample from
    each. The preparation heuristic is assumed to return only
    trace-1 Hermitian operators.
    """
    
    def __init__(self,
            updater, basis, prep_heuristic_class,
            meas_heuristic_class, other_fields=None
    ):
        super(ProductHeuristic, self).__init__(updater, basis, other_fields)
        self._ph = prep_heuristic_class(updater, basis=basis)
        self._mh = meas_heuristic_class(updater, basis=basis)
        
    def _next_prepmeas(self):
        prep = self._ph._next_measurement().unit()
        meas = self._mh._next_measurement()
        
        return prep, meas
    
class BestOfKMetaheuristic(Heuristic):
    """
    Draws :math:`k` different state or tomography
    measurements, then selects the one that has the largest
    expected value under the action of the covariance superoperator
    for the current posterior.
    """
    def __init__(self, updater, base_heuristic, k=3, other_fields=None):
        self._up = updater
        self._base_heuristic = base_heuristic
        self._k = k
        # TODO: consolidate this other_fields madness into
        #       a base heuristic class, not even just a tomography one!!
        self._other_fields = {} if other_fields is None else other_fields
        
    def __call__(self):
        expparams = np.array([
            self._base_heuristic()[0] for _ in range(self._k)
        ], dtype=self._up.model.expparams_dtype)
        meas = expparams['meas']
        cov_expectations = np.einsum('ei,ij,ej->e',
            meas, self._up.est_covariance_mtx(), meas
        )
        
        expparams = expparams[np.argmax(cov_expectations), None]

        for field, value in self._other_fields.items():
                expparams[field] = value
                
        return expparams

   