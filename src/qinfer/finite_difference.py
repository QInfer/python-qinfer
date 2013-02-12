#!/usr/bin/python

## FEATURES ##

from __future__ import division

## IMPORTS ##

import numpy as np

## CLASSES ##

class FiniteDifference(object):
    # TODO: add order parameter to generalize to higher orders.
    def __init__(self, func, n_args, h=1e-10):
        """
        Calculates finite differences of a scalar function of multiple
        variables.
        
        :param func: Function to take finite differences of.
        :type func: Function taking a single argument, an array of shape
            ``(n_points, n_args)``, and returning an array of shape
            ``(n_points,)``.
        :param n_args: Number of arguments represented by ``func``.
        :type n_args: int
        :param h: Step sizes to be used in calculating finite differences.
        :type h: Scalar, or array of shape ``(n_args,)``.
        """
        self.func = func
        self.n_args = n_args
        if np.isscalar(h): 
            self.h = h * np.ones((n_args,))
        else:
            self.h = h
        
    def central(self, xs):
        grad = np.zeros((self.n_args,))
        f = self.func
        
        for idx_arg in xrange(self.n_args):
            step = np.zeros((self.n_args,))
            step[idx_arg] = self.h[idx_arg]
            grad[idx_arg] = f(xs + step / 2) - f(xs - step / 2)
            
        return grad / self.h
        
    __call__ = central
    
