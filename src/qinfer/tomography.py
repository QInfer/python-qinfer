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

## FEATURES ####################################################################

from __future__ import division

## IMPORTS #####################################################################

import numpy as np
from abstract_model import Model
import scipy.linalg as la

## CLASSES #####################################################################

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
        return [('nqubits', 'int'), ('boolf', 'object')]
    
    @property
    def is_n_outcomes_constant(self):
        return True
    
    ## METHODS ##
    
    @staticmethod
    def are_models_valid(modelparams):
        return np.logical_and(modelparams.all(axis=1) >= 0,modelparams.all(axis=1) <= 1)
    
    def n_outcomes(self, expparams):
        return 2
    
    def likelihood(self, outcomes, modelparams, expparams):
        #unpack m and f
        m = expparams['nqubits']
        f = expparams['boolf'][0]
        
        #the first and last m bits     
        F0  = f[:2**m]        
        F1  = f[-2**m:]        

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
            #unpack m and f
            m = expparams['nqubits']
            f = expparams['boolf'][0]
            #the first and last m bits     
            F0  = f[:2**m]        
            F1  = f[-2**m:]
            
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
        
## TESTING CODE ################################################################

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
