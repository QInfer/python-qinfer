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
    def sample(self, n=1):
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
        
    def sample(self, n=1):
        shape = (self._n_rvs,n)# if n == 1 else (self._n_rvs, n)
        z = np.random.random(shape)
        return self._ranges[:, 0] + z * self._delta

    def grad_log_pdf(self, var):
        # TODO: This is not quite true
        return np.zeros(var.shape)