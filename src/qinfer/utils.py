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

###############################################################################

#TODO: not sure the best place to put this (The code seems to be floating
# around the internet with varying levels of copyrights <= MIT license)
def gammaln(n):
    
    # Check if we have a single outcome or an array.
    if not isinstance(n, np.ndarray):
        n = np.array([n])

    gln = np.zeros(n.shape[0])
    for ndx in xrange(n.shape[0]):    
        if n[ndx] < 1:
            return float('inf')
        if n[ndx] < 3:
            return 0.0
        c = [76.18009172947146, -86.50532032941677, \
             24.01409824083091, -1.231739572450155, \
             0.001208650973866179, -0.5395239384953 * 0.00001]
        x, y = float(n[ndx]), float(n[ndx])
        tm = x + 5.5
        tm -= (x + 0.5) * np.log(tm)
        se = 1.0000000000000190015
        for j in range(6):
            y += 1.0
            se += c[j] / y
        gln[ndx] = -tm + np.log(2.5066282746310005 * se / x)
    return gln
    
#TODO: cases for p=0 or p=1
def binomial_pdf(N,n,p):
    logprob = gammaln(N+1)-gammaln(n+1)- gammaln(N-n+1)  \
        + n*np.log(p)+(N-n)*np.log(1-p)
    return np.exp(logprob)