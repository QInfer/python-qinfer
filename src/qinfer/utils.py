#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# utils.py : some auxiliary functions
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

## IMPORTS ####################################################################

from __future__ import division
import numpy as np
from scipy.special import gammaln

###############################################################################

#TODO: cases for p=0 or p=1
def binomial_pdf(N,n,p):
    logprob = gammaln(N+1)-gammaln(n+1)- gammaln(N-n+1)  \
        + n*np.log(p)+(N-n)*np.log(1-p)
    return np.exp(logprob)

def outer_product(vec):        
    return (
        np.dot(vec[:, np.newaxis], vec[np.newaxis, :])
        if len(vec.shape) == 1 else
        np.dot(vec, vec.T)
        )
        
def particle_meanfn(weights, locations, fn):
    fn_vals = fn(locations).flatten()
    return np.sum(weights * fn_vals)
    
def particle_covariance_mtx(weights,locations):
        
        xs = locations.transpose([1, 0])
        ws = weights
        
        mu = np.sum(ws * xs, axis = 1)
        
        return (
            np.sum(
                ws * xs[:, np.newaxis, :] * xs[np.newaxis, :, :],
                axis=2
                )
            ) - np.dot(mu[..., np.newaxis], mu[np.newaxis, ...])