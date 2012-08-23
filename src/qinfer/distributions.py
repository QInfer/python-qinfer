#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# distributions.py: module for probability distributions
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

## IMPORTS #####################################################################

import numpy as np
import abc

## CLASSES #####################################################################

class Distribution(object):
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def sample(self):
        pass

class UniformDistribution(Distribution):
    def __init__(self, ranges=np.array([[0, 1]])):
        if not isinstance(ranges, np.ndarray):
            ranges = np.array(ranges)
            
        if len(ranges.shape) == 1:
            ranges = ranges[np.newaxis, ...]
    
        self._ranges = ranges
        self._n_rvs = ranges.shape[0]
        self._delta = ranges[:, 1] - ranges[:, 0]
        
    def sample(self):
        z = np.random.random((self._n_rvs,))
        return self._ranges[:, 0] + z * self._delta

# TODO: make the following into Distributions.

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
