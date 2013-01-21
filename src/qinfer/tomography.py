#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# SMC.py: Tomgraphic models module
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

## IMPORTS ##


from __future__ import division
import numpy as np
from abstract_model import Model
import scipy.linalg as la

class HaarUniform(object):
    """
    Creates a new Haar uniform prior on state space of dimension dim

    Parameters
    -----------
    dim : int
        dimension of the state space
    """
    def __init__(self,dim = 2):
        self.dim = dim
    
    def sample(self):
        #Generate random unitary (see e.g. http://arxiv.org/abs/math-ph/0609050v2)        
        z = (np.random.randn(self.dim,self.dim) + 1j*np.random.randn(self.dim,self.dim))/np.sqrt(2.0)
        q,r = la.qr(z)
        d = np.diag(r)
        
        ph = d/np.abs(d)
        ph = np.diag(ph)
        
        U = np.dot(q,ph)
        
        #TODO: generalize this to general dimensions
        #Apply Haar random unitary to |0> state to get random pure state
        psi = np.dot(U,np.array([1,0]))
        z = np.real(np.dot(psi.conj(),np.dot(np.array([[1,0],[0,-1]]),psi)))
        y = np.real(np.dot(psi.conj(),np.dot(np.array([[0,-1j],[1j,0]]),psi)))
        x = np.real(np.dot(psi.conj(),np.dot(np.array([[0,1],[1,0]]),psi)))
        
        return np.array([x,y,z])

class GinibreUniform(object):
    """
    Creates a prior on state space of dimension dim according to the Ginibre
    ensemble with parameter k
    see e.g. http://www.iitis.pl/~miszczak/files/papers/miszczak12generating
    
    Parameters
    -----------
    dim : int
        dimension of the state space
    """
    def __init__(self,dim = 2, k = 2):
        self.dim = dim
        self.k = k
        
    def sample(self):
        #Generate random matrix        
        z = np.random.randn(self.dim,self.k) + 1j*np.random.randn(self.dim,self.k)
        
        rho = np.dot(z,z.conj().transpose())
        rho = rho/np.trace(rho)
        
        z = np.real(np.trace(np.dot(rho,np.array([[1,0],[0,-1]]))))
        y = np.real(np.trace(np.dot(rho,np.array([[0,-1j],[1j,0]]))))
        x = np.real(np.trace(np.dot(rho,np.array([[0,1],[1,0]]))))
        
        return np.array([x,y,z])
                
# TODO: make the following into Distributions.        
class HilbertSchmidtUniform(object):
    """
    Creates a new Hilber-Schmidt uniform prior on state space of dimension dim
    see e.g. http://www.iitis.pl/~miszczak/files/papers/miszczak12generating

    Parameters
    -----------
    dim : int
        dimension of the state space
    """
    def __init__(self,dim = 2):
        self.dim = dim

    def sample(self):
        #Generate random unitary (see e.g. http://arxiv.org/abs/math-ph/0609050v2)        
        g = (np.random.randn(self.dim,self.dim) + 1j*np.random.randn(self.dim,self.dim))/np.sqrt(2.0)
        q,r = la.qr(g)
        d = np.diag(r)
        
        ph = d/np.abs(d)
        ph = np.diag(ph)
        
        U = np.dot(q,ph)

        #Generate random matrix        
        z = np.random.randn(self.dim,self.dim) + 1j*np.random.randn(self.dim,self.dim)
        
        rho = np.dot(np.dot(np.identity(self.dim)+U,np.dot(z,z.conj().transpose())),np.identity(self.dim)+U.conj().transpose())
        rho = rho/np.trace(rho)
        
        z = np.real(np.trace(np.dot(rho,np.array([[1,0],[0,-1]]))))
        y = np.real(np.trace(np.dot(rho,np.array([[0,-1j],[1j,0]]))))
        x = np.real(np.trace(np.dot(rho,np.array([[0,1],[1,0]]))))
        
        return np.array([x,y,z])

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
        
        pr0 = np.zeros((modelparams.shape[0], expparams.shape[0]))
        
        # Note that expparams['axis'] has shape (n_exp, 3).
        pr0 = 0.5*(1 + np.sum(modelparams*expparams['axis'],1))
        
        # Note that expparams['vis'] has shape (n_exp, ).
        pr0 = expparams['vis'] * pr0 + (1 - expparams['vis']) * 0.5

        pr0 = pr0[:,np.newaxis]
        
        # Now we concatenate over outcomes.
        return Model.pr0_to_likelihood_array(outcomes, pr0)        

class HTCircuitModel(Model):
    
    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return 1
        
    @property
    def expparams_dtype(self):
        return [('nqubits', 'int'), ('boolf', 'list')]
    
    @property
    def is_n_outcomes_constant(self):
        return True
    
    ## METHODS ##
    
    @staticmethod
    def are_models_valid(modelparams):
        return np.logical_and(modelparams.all(axis=1) >= 0,modelparams.all(axis=1) <= 1)
    
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
        #unpack m and f
        m = expparams['nqubits']
        f = expparams['boolf']
        
        # count the number of times the last bit of f is 0
        count = np.sum([bin(x)[-1] == '0' for x in f])      
        
        #probability of getting 0
        pr0 = 0.25*(1+modelparams)+0.5*(1-modelparams)*count/(2**m)
        
        #concatenate over outcomes
        return Model.pr0_to_likelihood_array(outcomes, pr0)
    
    def simulate_experiment(self, modelparams, expparams, repeat=1):
        #unpack m and f
        m = expparams['nqubits']
        f = expparams['boolf']
        
        # generate a random m-bit number
        x = np.random.randint(0,2**m,repeat)
        
        # set the outcome as the last bit of f(x)
        outcomes = []
        [outcomes.append(int(bin(f[d])[-1])) for d in x]
        return outcomes
        
## TESTING CODE ################################################################

if __name__ == "__main__":
    m = 8
    n = 10
    fn = np.arange(2**n)
    f  = fn[-2**(m):]
    
    param = np.array([[0]])
    expp = {'nqubits':m,'boolf':f} 
    

    model = HTCircuitModel()


    data = model.simulate_experiment(param,expp,10)
    
    print data

    L = model.likelihood(
        np.array(data),
        np.array([[0]]),
        expp
    )
    print L

#### TEST PRIORS #############################################################        
#    from mpl_toolkits.mplot3d import Axes3D
#    import matplotlib.pyplot as plt 
#
#    prior = HilbertSchmidtUniform()
#    
#    n = 1000
#    x = np.zeros((n,))   
#    y = np.zeros((n,))    
#    z = np.zeros((n,))
#    
#    for idx in xrange(n):
#        temp = prior.sample()
#        x[idx] = temp[0]        
#        y[idx] = temp[1]        
#        z[idx] = temp[2]        
#        
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    
#    ax.scatter(x,y,z)
#    plt.show()
