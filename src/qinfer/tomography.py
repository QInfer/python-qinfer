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

import numpy as np #CF: does this create uneessary overhead?

class StateModel(object):
    """
    Represents an experimental system with unknown quantum state,
    and known measurement operators.
    """    
    
    def __init__(self, hs_dim):
        self.hs_dim = hs_dim
        
    def likelihood(data,state):
        """
        Calculates the likelihood function at the point state.
        This is given by the Born rule and is the probability of
        data given state.
        
        Parameters
        ----------
        data = (TODO:some data structure)
            data structure containing measurement data
        state = ndarray
            quantum state in the computational basis
        """
        
        meas_ops,counts = method to extract effects of POVM and counts (data)
        
        prob = some function for multinomial PDF(counts,p)
                
        return prob