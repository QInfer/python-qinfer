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
from utils import gammaln
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
        
        return np.dot(q,ph)
        
                
# TODO: make the following into Distributions.        
class HilbertSchmidtUniform(object):
    """
    Creates a new Hilber-Schmidt uniform prior on state space of dimension dim

    Parameters
    -----------
    dim : int
        dimension of the state space
    """
    def __init__(self,dim = 2):
        self.dim = dim


class QubitStatePauliModel(Model):
    """
    Represents an experimental system with unknown quantum state,
    and Pauli measurement operators.
    """    
    
    @property
    def n_modelparams(self):
        return 3
        
    @property
    def expparams_dtype(self):
        return 'int'

    @staticmethod
    def is_model_valid(self, modelparams):
        return modelparams[0]**2 + modelparams[1]**2 + modelparams[2]**2 <= 1
    
    def n_outcomes(self, expparams):
        return expparams[0]
        
    def likelihood(self, outcomes, modelparams, expparams):
        """
        Calculates the likelihood function at the states specified 
        by modelparams and measurement specified by expparams.
        This is given by the Born rule and is the probability of
        outcomes given the state and measurement operators.
        
        Parameters
        ----------
        outcomes = 
            measurement outcome counts
        expparams = 
            number of measurements
        modelparams = 
            quantum state Bloch vector
        """
        
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(QubitStatePauliModel, self).likelihood(outcomes, modelparams, expparams)
        
        ps = np.zeros((3,))
        
        ps[0] = 0.5*(1+modelparams[0])
        ps[1] = 0.5*(1+modelparams[1])
        ps[2] = 0.5*(1+modelparams[2])

        logprob = 3*gammaln(expparams[0]+1) \
        -gammaln(outcomes[0]+1) - gammaln(expparams[0] - outcomes[0]+1) \
        -gammaln(outcomes[1]+1) - gammaln(expparams[0] - outcomes[1]+1) \
        -gammaln(outcomes[2]+1) - gammaln(expparams[0] - outcomes[2]+1) \
        +outcomes[0]*np.log(ps[0]) + (expparams[0] - outcomes[0])*np.log(1-ps[0]) \
        +outcomes[1]*np.log(ps[1]) + (expparams[0] - outcomes[1])*np.log(1-ps[1]) \
        +outcomes[2]*np.log(ps[2]) + (expparams[0] - outcomes[2])*np.log(1-ps[2]) \
                
        return np.exp(logprob)
        
## TESTING CODE ################################################################

if __name__ == "__main__":
#
#    
#    # commented out stuff below is for 3D
##    from mpl_toolkits.mplot3d import Axes3D
#    import matplotlib.pyplot as plt
#    
#    
#    m = QubitStatePauliModel()
#    
#    fig = plt.figure()
##    ax = fig.add_subplot(111, projection='3d')
#    x = y = np.arange(-1, 1, 0.01)
#    X, Y = np.meshgrid(x, y)
#    zs = np.array([m.likelihood(
#    np.array([25,1,25]),
#    np.array([x,y,0]),
#    np.array([50]))
#    for x,y in zip(np.ravel(X), np.ravel(Y))])
#    Z = zs.reshape(X.shape)
#  
#    zs2 = np.array([m.likelihood(
#    np.array([25,25,25]),
#    np.array([x,y,0]),
#    np.array([50]))
#    for x,y in zip(np.ravel(X), np.ravel(Y))])
#    Z2 = zs2.reshape(X.shape)
#    
#    zs3 = np.array([m.likelihood(
#    np.array([8,8,25]),
#    np.array([x,y,0]),
#    np.array([50]))
#    for x,y in zip(np.ravel(X), np.ravel(Y))])
#    Z3 = zs3.reshape(X.shape)
#    
#  #  ax.plot_surface(X, Y, Z)
#  #  ax.set_xlabel('X')
#  #  ax.set_ylabel('Y')
#  #  ax.set_zlabel('Pr(data|X,Y,0)')
#
#    t = np.arange(0, 2*np.pi, 0.01)
#    xx = np.cos(t)
#    yy = np.sin(t)    
#    
#    plt.plot(xx,yy,'k')
#
#    plt.contour(X,Y,Z)
#    plt.contour(X,Y,Z2)
#    plt.contour(X,Y,Z3)
#    
#    
#    plt.show()    


#### TEST PRIORS #############################################################        
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt 

    prior = HaarUniform()
    
    n = 1000
    x = np.zeros((n,))   
    y = np.zeros((n,))    
    z = np.zeros((n,))
    
    for idx in xrange(n):
        U = prior.sample()
        psi = np.dot(U,np.array([1,0]))
        z[idx] = np.real(np.dot(psi.conj(),np.dot(np.array([[1,0],[0,-1]]),psi)))
        y[idx] = np.real(np.dot(psi.conj(),np.dot(np.array([[0,-1j],[1j,0]]),psi)))
        x[idx] = np.real(np.dot(psi.conj(),np.dot(np.array([[0,1],[1,0]]),psi)))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(x,y,z)
    plt.show()