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

class StateModel(object):
    """
    Represents an experimental system with unknown quantum state,
    and known measurement operators.
    """    
    
    def __init__(self, hs_dim):
        self.hs_dim = hs_dim
        
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
        
        ps = method to turn params to probabilities(expparams,modelparams)                
        prob = some function for multinomial PDF(outcomes,ps)
                
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
        ps = zeros((,3))
        
        return ps