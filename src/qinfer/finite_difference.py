#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# finite_difference.py: Implementation of central finite difference
#     approximator for first derivatives.
##
# © 2017, Chris Ferrie (csferrie@gmail.com) and
#         Christopher Granade (cgranade@cgranade.com).
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the copyright holder nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
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
    
