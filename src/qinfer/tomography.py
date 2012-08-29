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

    m = QubitStatePauliModel()
    L = m.likelihood(
        np.array([5,5,5]),
        np.array([0,0,0]),
        np.array([10])
    )
    print L