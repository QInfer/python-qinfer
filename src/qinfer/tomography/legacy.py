#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# SMC.py: Tomgraphic models module
##
# © 2012 Chris Ferrie (csferrie@gmail.com) and
#        Christopher E. Granade (cgranade@gmail.com).
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

## TODO #######################################################################
# - Refactor modelparams <-> Qobj into common functions that memoize over
#   bases for each nq.
# - Write QST, QPT models using new distributions.
# - Deprecate old functionality.
# - Write unit tests for new mps <-> Qobj conversions.

## FEATURES ###################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

## IMPORTS ####################################################################

import numpy as np
import scipy.linalg as la

from qinfer.abstract_model import Model
from qinfer.distributions import Distribution, SingleSampleMixin

from qinfer.utils import get_qutip_module
qt = get_qutip_module('3.2')

## FUNCTIONS #################################################################

def require_qutip():
    if qt is None:
        raise ImportError("QuTiP version 3.1.0 or later required.")

def conjugate(B, X):
    return np.dot(B, np.dot(X, B.conj().transpose()))

def _qubit_superdims(nq):
    ds = [2] * nq
    return [[ds, ds]] * 2

## CLASSES ####################################################################

class GinibreQubitDistribution(SingleSampleMixin, Distribution):
    """
    Creates a new uniform prior over density operators on :math:`n` qubits,
    using the rank-:math:`k` Ginibre distribution. Sampled density operators
    are represented by vectors in the Pauli basis excluding the traceful
    element.
    
    This class is implemented using QuTiP (v3.1.0 or later), and thus will not
    work unless QuTiP is installed.


    :param int nq: Number of qubits.
    :param int rank: Rank of density operators to be sampled. If ``None``,
        full-rank density operators are sampled.
    """
    
    def __init__(self, nq=1, rank=None):
        require_qutip()
        self._nq = nq
        self._rank = rank
        self._dim = 2**nq

        # This arcane line takes the vectorized Pauli basis transformation
        # used by QuTiP and converts into a form that's useful for us.
        # Notably an array (not a matrix!) that excludes the traceful parts.
        self._paulis = qt.visualization._pauli_basis(nq).data.todense().H[1:, :].view(np.ndarray)

    @property
    def n_rvs(self):
        return self._dim ** 2 - 1
        
    def _sample(self):
        # Generate and flatten a density operator, so that we can multiply it
        # by the transformation defined above.
        rho = qt.rand_dm_ginibre(self._dim, self._rank).data.todense().view(np.ndarray).flatten(order='C')
        return np.real(np.dot(self._paulis, rho))

class BCSZQubitDistribution(SingleSampleMixin, Distribution):
    """
    Represents the BCSZ prior over CPTP maps of a given Choi (Kraus) rank.
    The returned model parameters are a vectorization of a supermatrix
    in the Pauli basis, with the first row omitted.

    :param int nq: Number of qubits on which random variate CPTP maps act.
    :param int rank: Choi/Kraus rank of the random variate maps. For
        ``rank = None``, full-rank will be assumed, such that the BCSZ
        prior is the projection of the Hilbert-Schmidt prior onto the
        CPTP constraint.
    :param bool enforce_tp: If ``False``, relaxes the constraint that
        sampled channels are trace-preserving.
    """
    def __init__(self, nq=1, rank=None, enforce_tp=True):
        require_qutip()
        self._nq = nq
        self._basis = (
            qt.visualization._pauli_basis(nq) / np.sqrt(2**nq)
        ).data.todense().H.view(np.ndarray)
        self._dims = 2**nq
        self._superdims = 4**nq
        self._rank = rank
        self._enforce_tp = enforce_tp

    @property
    def n_rvs(self):
        # The top "row" of the superoperator in the Pauli basis is completely
        # determined by the assumption of a CPTP map (S_{0i} = delta_{0i}),
        # and so we strip it off, removing d² params.
        return self._superdims ** 2 - self._superdims

    def _sample(self):
        # Since the dense representation of S is a matrix and not an array,
        # flattening in FORTRAN order gives MATLAB-like behavior (implicitly
        # two-index). Thus, we need to index away the first axis.
        return np.real(
            conjugate(self._basis,
                qt.rand_super_bcsz(
                    self._dims, enforce_tp=self._enforce_tp, rank=self._rank
                ).data.todense().view(np.ndarray)
            )
        )[1:, :].flatten(order='F')

    def to_qobj(self, modelparams):
        return [
            qt.Qobj(
                conjugate(
                    self._basis.conj().transpose(),
                     np.vstack((
                         np.hstack((
                             [1], np.zeros((self._superdims - 1, ))
                         )),
                        modelparams.reshape((self._superdims - 1, self._superdims), order='F')
                    ))
                ),
                dims=_qubit_superdims(self._nq)
            )
            for x in modelparams
        ]

class QubitStatePauliModel(Model):
    """
    Represents an experimental system with unknown quantum state,
    and a limited visibility projective measurement.
    
    States are represented in the Pauli representation.
    """    
    
    @property
    def n_modelparams(self):
        return 3
        
    @property
    def expparams_dtype(self):
        # return 'float' <---- This implies a two-index array of scalars,
        #                      but we need a one-index array of records.
        return [('axis', '3f4'), ('vis', 'float')]
        #                 ^
        #                 |
        #                 3 floats, each four bytes wide

    @staticmethod
    def are_models_valid(modelparams):
        return modelparams[:, 0]**2 + modelparams[:, 1]**2 + modelparams[:, 2]**2 <= 1
    
    def n_outcomes(self, expparams):
        return 2
        
    def likelihood(self, outcomes, modelparams, expparams):
        """
        Calculates the likelihood function at the states specified 
        by modelparams and measurement specified by expparams.
        This is given by the Born rule and is the probability of
        outcomes given the state and measurement operator.
        
        Parameters
        ----------
        outcomes = 
            measurement outcome
        expparams = 
            Bloch vector of measurement axis and visibility
        modelparams = 
            quantum state Bloch vector
        """
        
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(QubitStatePauliModel, self).likelihood(outcomes, modelparams, expparams)
        
        # Note that expparams['axis'] has shape (n_exp, 3).
        pr0 = 0.5*(1 + np.sum(modelparams*expparams['axis'],1))
        
        # Note that expparams['vis'] has shape (n_exp, ).
        pr0 = expparams['vis'] * pr0 + (1 - expparams['vis']) * 0.5

        pr0 = pr0[:,np.newaxis]

        # Now we concatenate over outcomes.
        return Model.pr0_to_likelihood_array(outcomes, pr0)        

class RebitStatePauliModel(Model):
    """
    Represents an experimental system with unknown quantum state restricted
    to the X-Y in the Bloch sphere.
    """    
    
    @property
    def n_modelparams(self):
        return 2
        
    @property
    def expparams_dtype(self):
        return [('axis', '2f4'), ('vis', 'float')]

    @staticmethod
    def are_models_valid(modelparams):
        return True#modelparams[:, 0]**2 + modelparams[:, 1]**2 <= 1
    
    def n_outcomes(self, expparams):
        return 2
        
    def likelihood(self, outcomes, modelparams, expparams):
        """
        Calculates the likelihood function at the states specified 
        by modelparams and measurement specified by expparams.
        This is given by the Born rule and is the probability of
        outcomes given the state and measurement operator.
        
        Parameters
        ----------
        outcomes = 
            measurement outcome
        expparams = 
            Bloch vector of measurement axis
        modelparams = 
            quantum state Bloch vector
        """
        
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(RebitStatePauliModel, self).likelihood(outcomes, modelparams, expparams)
        
        pr0 = np.zeros((modelparams.shape[0], expparams.shape[0]))
        
        # Note that expparams['axis'] has shape (n_exp, 3).
        pr0 = 0.5*(1 + np.sum(modelparams*expparams['axis'],1))
        
        # Use the following hack if you don't want to ensure positive weights
        pr0[pr0 < 0] = 0
        pr0[pr0 > 1] = 1
        
        pr0 = pr0[:,np.newaxis]
        
        # Now we concatenate over outcomes.
        return Model.pr0_to_likelihood_array(outcomes, pr0)       


class MultiQubitStatePauliModel(Model):
    """
    Represents an experimental system with unknown quantum state,
    and a limited visibility projective measurement.
    
    States are represented in the Pauli representation.
    """    
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        super(MultiQubitStatePauliModel, self).__init__()

        
    @property
    def n_modelparams(self):
        return 4**self.n_qubits - 1
        
    @property
    def expparams_dtype(self):
        return [('pauli', 'int'), ('vis', 'float')]

    @staticmethod
    def are_models_valid(modelparams):
        return True
    
    def n_outcomes(self, expparams):
        return 2
        
    def likelihood(self, outcomes, modelparams, expparams):
        """
        Calculates the likelihood function at the states specified 
        by modelparams and measurement specified by expparams.
        This is given by the Born rule and is the probability of
        outcomes given the state and measurement operator.
        """
        
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(MultiQubitStatePauliModel, self).likelihood(outcomes, modelparams, expparams)
        
        
        # Note that expparams['axis'] has shape (n_exp, 3).
        pr0 = 0.5*(1 + modelparams[:,expparams['pauli']])

        # Use the following hack if you don't want to ensure positive weights
        pr0[pr0 < 0] = 0
        pr0[pr0 > 1] = 1
        
        # Note that expparams['vis'] has shape (n_exp, ).
        pr0 = expparams['vis'] * pr0 + (1 - expparams['vis']) * 0.5

        # Now we concatenate over outcomes.
        return Model.pr0_to_likelihood_array(outcomes, pr0)        
        
class HTCircuitModel(Model):
    
    def __init__(self, n_qubits, n_had, f):
        self.n_qubits = n_qubits
        self.n_had    = n_had
        self.f        = f
        super(HTCircuitModel, self).__init__()
    
    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return 1
        
    @property
    def expparams_dtype(self):
        return [('null', 'int')]
    
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
        m = self.n_had
        
        #the first and last m bits     
        F0  = self.f[:2**m]        
        F1  = self.f[-2**m:]        

        # count the number of times the last bit of F is 0
        count0 = np.sum((F0+1) % 2)      
        count1 = np.sum((F1+1) % 2)      
        
        #probability of getting 0
        pr0 = modelparams*count0/(2**m)+(1-modelparams)*count1/(2**m)
        
        #concatenate over outcomes
        return Model.pr0_to_likelihood_array(outcomes, pr0)
    
    def simulate_experiment(self, modelparams, expparams, repeat=1, use_like = False):
        if use_like:
            return super(HTCircuitModel,self).simulate_experiment(modelparams, expparams, repeat)
        else:
            m = self.n_had
            
            #the first and last m bits     
            F0  = self.f[:2**m]        
            F1  = self.f[-2**m:]        
                
            outcomes = np.zeros([repeat,modelparams.shape[0],expparams.shape[0]])
            #select |0> or |1> with probability given by lambda
            idx_zeros = np.random.random([repeat,modelparams.shape[0],expparams.shape[0]]) > 0.5*(1-modelparams)
            num_zeros = np.sum(idx_zeros)
            num_ones  = modelparams.shape[0]*repeat*expparams.shape[0] - num_zeros 

            # for the |0> state set the outcomes to be the last bit of F0(x)
            x = np.random.randint(0,2**m,num_zeros)            
            outcomes[idx_zeros] = np.mod(F0[x],2)
            
            # for the |1> state set the outcomes to be the last bit of F1(x)
            x = np.random.randint(0,2**m,num_ones)
            outcomes[np.logical_not(idx_zeros)] = np.mod(F1[x],2)
            
            return outcomes
        
## TESTING CODE ###############################################################

# TODO: move to examples/.
if __name__ == "__main__":
    m = 2
    n = 4
    fn = np.arange(2**n)
    
    param = np.array([[0],[1]])
    expp = {'nqubits':m,'boolf':fn} 
    

    model = HTCircuitModel()


    data = model.simulate_experiment(param,expp,repeat = 4,use_like=False)
    
    L = model.likelihood(
        np.array([0,1,0,1]),
        np.array([[0]]),
        expp
    )
    print(L)

