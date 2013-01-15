#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# metrics.py: Metrics for use with SciPy and sklearn functions.
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

## FEATURES ####################################################################

from __future__ import division

## ALL #########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
    'weighted_pairwise_distances'
]

## IMPORTS #####################################################################

import numpy as np
import scipy.linalg as la
import warnings

from utils import outer_product

try:
    import sklearn
    import sklearn.metrics
    import sklearn.metrics.pairwise
except ImportError:
    warnings.warn("Could not import scikit-learn. Some features may not work.")
    sklearn = None

## FUNCTIONS ###################################################################

def weighted_pairwise_distances(X, w, metric='euclidean'):
    r"""
    Given a feature matrix ``X`` with weights ``w``, calculates the modified
    distance metric :math:`\tilde{d}(p, q) = d(p, q) / (w(p) w(q) N^2)`, where
    :math:`N` is the length of ``X``. This metric is such that "heavy" feature
    vectors are considered to be closer to each other than "light" feature
    vectors, and are hence correspondingly less likely to be considered part of
    the same cluster.
    """
    
    if sklearn is None:
        raise ImportError("This function requires scikit-learn.")
    
    base_metric = sklearn.metrics.pairwise.pairwise_distances(X, metric=metric)
    N = w.shape[0]
    w_matrix = outer_product(w) * N**2
    
    return base_metric / w_matrix
    
