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

from __future__ import division

## IMPORTS ###################################################################

from qinfer import Heuristic
from qinfer.tomography.bases import pauli_basis

import numpy as np

from abc import abstractmethod

# Since the rest of QInfer does not require QuTiP,
# we need to import it in a way that we don't propagate exceptions if QuTiP
# is missing or is too early a version.
try:
    import qutip as qt
    from distutils.version import LooseVersion
    _qt_version = LooseVersion(qt.version.version)
    if _qt_version < LooseVersion('3.1'):
        qt = None
except ImportError:
    qt = None

## EXPORTS ###################################################################
# TODO

## FUNCTIONS #################################################################

# TODO: document, contribute to QuTiP?
def heisenberg_weyl_operators(d=2):
    w = np.exp(2 * np.pi * 1j / d)
    X = qt.Qobj([
        qt.basis(d, (idx + 1) % d).data.todense().view(np.ndarray)[:, 0] for idx in xrange(d)
    ])
    Z = qt.Qobj(np.diag(w ** np.arange(d)))
    
    return [X**i * Z**j for i in xrange(d) for j in xrange(d)]

## CLASSES ####################################################################

class StateTomographyHeuristic(Heuristic):
    # TODO: docstring
    def __init__(self, updater, other_fields=None):        
        self._up = updater
        self._other_fields = {} if other_fields is None else other_fields
        self._dim = updater.model.base_model.dim
        self._basis = updater.model.base_model.basis
        
    def __call__(self):
        expparams = np.zeros((1,), dtype=self._up.model.expparams_dtype)
        expparams['meas'][0, :] = self._basis.state_to_modelparams(
            self._next_measurement()
        )

        for field, value in self._other_fields.iteritems():
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
    
    def __init__(self, updater, other_fields=None):
        super(RandomStabilizerStateHeuristic, self).__init__(
            updater, other_fields
        )
    
        self._hw_group = heisenberg_weyl_operators(self._dim)
    
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
    def __init__(self, updater, other_fields=None):
        super(RandomStabilizerStateHeuristic, self).__init__(
            updater, other_fields
        )
        
        nq = int(np.log2(self._dim))
        if 2**nq != self._dim:
            raise ValueError("RandomPauliHeuristic is defined only for qubits.")
        
        self._nq = nq
        self._pauli_basis = pauli_basis(nq)
        
    def _next_measurement(self):
        return np.random.choice(self._pauli_basis)

class BestOfKMetaheuristic(Heuristic):
    """
    Draws :math:`k` different state or tomography
    measurements, then selects the one that has the largest
    expected value under the action of the covariance superoperator
    for the current posterior.
    """
    def __init__(self, updater, base_heuristic, k=3):
        self._up = updater
        self._base_heuristic = base_heuristic
        self._k = k
        
    def __call__(self):
        expparams = np.array([
            self._base_heuristic() for _ in xrange(self._k)
        ], dtype=self._up.model.expparams_dtype)
        meas = expparams['meas']
        cov_expectations = np.dot(meas, self._up.est_covariance_mtx(), meas.T)
        
        return expparams[np.argmax(cov_expectations), None, :]
