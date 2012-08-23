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

import numpy as np
from abstract_model import *

class QubitStatePauliModel(Model):
    """
    Represents an experimental system with unknown quantum state,
    and Pauli measurement operators.
    """    
    
    def __init__(self, hs_dim):
        Model.__init__(self,hs_dim=2)
        
    def likelihood(outcomes,expparams,modelparams):
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
            measurement operator specs and other experimental specs
        modelparams = 
            quantum state specs
        """
        
        # assumes Pauli X,Y,Z measurements for now (i.e. expparams does nothing)
        
        ps = self.params2probs(expparams,modelparams)                
        prob = self.finalprobs(outcomes,ps)
                
        return prob
        
    def params2probs(expparams,modelparams):
        """
        Converts (via the Born rule) a description of the states and
        measurements to probabilities
        
        Parameters
        ----------
        expparams = 
            measurement operator specs and other experimental specs
        modelparams = 
            quantum state specs
        """        
        
        # assumes Pauli X,Y,Z measurements for now (i.e. expparams does nothing)
        ps = np.zeros((3,))
        
        ps[0] = 0.5*(1+modelparams[0])
        ps[1] = 0.5*(1+modelparams[1])
        ps[2] = 0.5*(1+modelparams[2])
        
        return ps
        
    def finalprob(outcomes,ps):
        """
        Converts the probabilities of each measurement into the final 
        
        Parameters
        ----------
        outcomes = 
            measurement outcome counts
        ps = 
            probabilities for each measurement outcome
        """
        num_meas = np.sum(outcomes,axis=1)
        # assumes each measurement is Bernoulli trial
        logprob = term with log of factorials + 
                    outcomes[0,0]*log(ps[0]) + outcomes[0,1]*log(1-ps[0]) 
                    outcomes[1,0]*log(ps[1]) + outcomes[1,1]*log(1-ps[1])
                    outcomes[2,0]*log(ps[2]) + outcomes[2,1]*log(1-ps[2])
                
        return exp(logprob)