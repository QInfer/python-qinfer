#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# geometry.py: Computational geometry utilities.
##
# © 2016 Chris Ferrie (csferrie@gmail.com) and
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

## EXPORTS ###################################################################

__all__ = [
    # TODO
]

## IMPORTS ###################################################################

import numpy as np

from qinfer.utils import requires_optional_module

try:
    import cvxopt
    import cvxopt.solvers
except:
    cvxopt = None

## FUNCTIONS #################################################################

@requires_optional_module('cvxopt')
def convex_distance(test_point, convex_points):
    r"""
    Returns the distance :math:`d(\vec{x}, S) =
    \min_{\vec{z} \in \operatorname{conv}(S)} \| \vec{x} - \vec{z} \|^2` from
    a point :math:`\vec{x}` to the convex closure of a set of points
    :math:`S`.
    """
    # TODO: document this much better.

    n_points, dimension = convex_points.shape

    # Start by centering the convex region of interest.
    # NB: don't use in-place (-=) here, since we want a copy.
    convex_points = convex_points - test_point

    # Construct the quadratic program parameters P and q required by CVXOPT.
    # Since we want min x^T P x, with x representing the weight of each point
    # in our set of a convex combination, we need to construct the outer
    # product
    #     P_{ik} = sum_j y_ij y_jk
    # where y_ij is the jth component of the ith point in our set.
    P = cvxopt.matrix(np.dot(convex_points, convex_points.T))
    # We have no linear objective, so we set q = 0.
    # Similarly, q is a vector of all ones representing the sum
    # over points in the set.
    q = cvxopt.matrix(np.zeros((n_points, 1)))
    # To enfoce x_i ≥ 0, we need G = -\mathbb{1} and h = 0,
    # such that CVXOPT's constraint Gx ≥ h is correct.
    G = cvxopt.matrix(-np.eye(n_points))
    h = cvxopt.matrix(0.0, (n_points, 1))

    # We do, however, have a linear constraint A x = 1.
    A = cvxopt.matrix(1.0, (1, n_points))
    b = cvxopt.matrix(1.0)

    # Now we solve with cvxopt.
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)

    # Return the obtained primal objective, noting that CVXOPT finds
    # min_x 1/2 x^T P X, so that we must multiply by 2, then take the
    # square root. Taking the square root, we have to be a little
    # careful, since finite tolerances can cause the objective to
    # be negative, even though we constrained it to be at least zero.
    # Thus, we take the max of the objective and zero.
    return np.sqrt(2 * max(0, solution['primal objective']))
