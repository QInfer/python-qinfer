#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# finite_difference.py: Implementation of central finite difference
#     approximator for first derivatives.
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

## FEATURES ###################################################################

from __future__ import absolute_import
from __future__ import division

## ALL ########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
    'FiniteDifference'
]

## IMPORTS ####################################################################

from builtins import range

import numpy as np

## CLASSES ####################################################################

class FiniteDifference(object):
    """
    Calculates finite differences of a scalar function of multiple
    variables.
        
    :param func: Function to take finite differences of.
    :type func: Function taking a single argument, an array of shape
        ``(n_points, n_args)``, and returning an array of shape
        ``(n_points,)``.
    :param int n_args: Number of arguments represented by ``func``.
    :param h: Step sizes to be used in calculating finite differences.
    :type h: Scalar, or array of shape ``(n_args,)``.
    """

    # TODO: add order parameter to generalize to higher orders.
    def __init__(self, func, n_args, h=1e-10):
        self.func = func
        self.n_args = n_args
        if np.isscalar(h): 
            self.h = h * np.ones((n_args,))
        else:
            self.h = h
        
    def central(self, xs):
        grad = np.zeros((self.n_args,))
        f = self.func
        
        for idx_arg in range(self.n_args):
            step = np.zeros((self.n_args,))
            step[idx_arg] = self.h[idx_arg]
            grad[idx_arg] = f(xs + step / 2) - f(xs - step / 2)
            
        return grad / self.h
        
    __call__ = central
    
