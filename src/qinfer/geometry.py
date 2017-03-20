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
    'convex_distance',
    'approximate_convex_hull'
]

## IMPORTS ###################################################################

import numpy as np
from scipy.spatial.distance import cdist

from qinfer.utils import requires_optional_module, uniquify

try:
    import cvxopt
    import cvxopt.solvers
except:
    cvxopt = None

## FUNCTIONS #################################################################

# FIXME: add outgoing citations.

@requires_optional_module('cvxopt')
def convex_distance(test_point, convex_points):
    r"""
    Returns the distance :math:`d(\vec{x}, S) =
    \min_{\vec{z} \in \operatorname{conv}(S)} \| \vec{x} - \vec{z} \|^2` from
    a point :math:`\vec{x}` to the convex closure of a set of points
    :math:`S`.
    """
    # TODO: document this much better, esp. params/returns.

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
    solution = cvxopt.solvers.qp(P, q, G, h, A, b,
        options={'show_progress': False}
    )

    # Return the obtained primal objective, noting that CVXOPT finds
    # min_x 1/2 x^T P X, so that we must multiply by 2, then take the
    # square root. Taking the square root, we have to be a little
    # careful, since finite tolerances can cause the objective to
    # be negative, even though we constrained it to be at least zero.
    # Thus, we take the max of the objective and zero.
    return np.sqrt(2 * max(0, solution['primal objective']))

def rand_argmin(arr, tol=1e-7):
    minimum = np.min(arr)
    return np.random.choice(np.nonzero(arr <= minimum + tol)[0])

def expand_hull_directed_search(partial_hull, candidate_points, interior_thresh=1e-7):
    r"""
    Uses the Sartipizadeh and Vincent 2016 algorithm to expand
    an approximate convex hull by finding a point :math:`\vec{x}`
    solving

    .. math::

        \vec{x} = \operatorname{arg\,min}_{\vec{x} \in S \setminus E}
                  \max_{\vec{z} \in S \setminus E}
                  d(\vec{z}, E \cup \{\vec{x}\})

    where :math:`E` is the set of points comprising the approximate
    hull so far, and where :math:`S` is the set of points that we
    wish to approximate the convex hull of. That is, this function
    finds a point to add to an approximate hull that minimizes
    the worst-case distance to any point not currently in the hull.
    """
    # TODO: document this much better, esp. params/returns.

    # returns:
    #    index of point to add to the hull
    #    current error of the approximate hull
    #    masked matrix of known convex distances

    # Our strategy will be to build up a matrix D_{i, j}
    # defined as:
    #
    #     Dᵢⱼ ≔ d(xᵢ, E ∪ {xⱼ})
    #
    # where the set of candidate points S \ E = {xᵢ} is represented
    # as the array candidate_points.
    #
    # NB: our notation is different from VC16, since they reuse
    # the letter E.

    n_partial, dimension = partial_hull.shape
    n_candidates, dimension_check = candidate_points.shape
    assert dimension == dimension_check

    # Start by allocating arrays to represent how far into
    # each maximization we've considered for the minimization over
    # candidates.
    n_distances_considered = np.ones((n_candidates,), dtype=int)

    # We will also need to remember the list of points interior
    # to each candidate so that they can be removed later.
    interior_points_found = [[] for idx in range(n_candidates)]

    # We then define how to compute each element.
    closest_miss = [np.inf]
    def distance(idx_max, idx_min):
        if idx_max == idx_min:
            this_distance = 0
        else:
            this_distance = convex_distance(
                candidate_points[idx_max],
                np.concatenate([
                    partial_hull,
                    candidate_points[idx_min, np.newaxis]
                ], axis=0)
            )
            if this_distance < closest_miss[0]:
                closest_miss[0] = this_distance

        if this_distance < interior_thresh:
            interior_points_found[idx_min].append(idx_max)
        return this_distance

    # Next, we allocate an array for the worst maxima we've seen for
    # each candidate.
    worst_distances = np.array([
        distance(0, idx_min)
        for idx_min in range(n_candidates)
    ])

    # Using this, we can find our best candidate so far
    idx_best_candidate = rand_argmin(worst_distances)

    # ITERATION: Look at each "row" in turn until we get to the bottom. At each step,
    #            evalaute the next element of the "column" with the best (smallest)
    #            maximum.
    while n_distances_considered[idx_best_candidate] < n_candidates:

        next_distance = distance(n_distances_considered[idx_best_candidate], idx_best_candidate)
        n_distances_considered[idx_best_candidate] += 1

        if next_distance > worst_distances[idx_best_candidate]:
            worst_distances[idx_best_candidate] = next_distance

        idx_best_candidate = rand_argmin(worst_distances)

    # COMPLETION: Return the best element we've found so far, along with the value
    #             at that element, and the indices of interior points for the newly
    #             expanded hull.
    approximation_error = worst_distances[idx_best_candidate]
    idxs_interior_points = interior_points_found[idx_best_candidate]

    print(closest_miss)

    return idx_best_candidate, approximation_error, idxs_interior_points


def approximate_convex_hull(points, max_n_vertices, desired_approximation_error,
        interior_thresh=1e-7,
        seed_method='extreme'
    ):
    
    n_points, dimension = points.shape
    max_n_vertices = min(n_points - 1, max_n_vertices)

    # FIXME: we should shuffle.

    # Start by picking a point a seed point to grow the hull from.
    if seed_method == 'extreme':
        # http://www.sciencedirect.com/science/article/pii/S2405896315009866
        # idx_dimension = np.random.randint(dimension)
        # sign = np.random.choice([-1, 1])
        idx_seed_points = uniqify([
            (sign * points[:, idx_dimension]).argmin()
            for idx_dimension in range(dimension)
            for sign in [-1, 1]
        ])

    elif seed_method == 'centroid':
        centroid = np.mean(points, axis=0)
        centroid_distances = cdist(points, centroid[None, :])[:, 0]
        idx_seed_points = [centroid_distances.argmin(), centroid_distances.argmax()]


    # We maintain masks to indicate which points are in the approximate
    # hull, and which masks are candidates (haven't been eliminated as
    # interior).
    hull_mask = np.zeros((n_points,), dtype=bool)
    hull_mask[idx_seed_points] = True

    candidate_mask = ~hull_mask.copy()

    # Before we iterate, remove candidates that are interior to the seed hull.
    # TODO

    # Iterate until we hit max_n_vertices or epsilon_desired, expanding
    # the hull each time and eliminating the interior points.
    while True:
        partial_hull = points[hull_mask]
        candidate_points = points[candidate_mask]
        idxs_partial_hull = np.nonzero(hull_mask)[0]

        idx_best_candidate, approximation_error, idxs_interior_points = \
            expand_hull_directed_search(partial_hull, candidate_points, interior_thresh=interior_thresh)

        # Add the new best candidate to the approximate hull,
        # and eliminate any candidate points interior to the new point.
        idxs_candidates = np.nonzero(candidate_mask)[0]
        hull_mask[idxs_candidates[idx_best_candidate]] = True
        candidate_mask[idxs_candidates[idxs_interior_points]] = False
        
        # Eliminate anything from the hull_mask
        # that's interior to the new point.
        n_hull_removed = 0
        reduced_hull_mask = hull_mask.copy()
        for idx_previous_hull_point in idxs_partial_hull:
            reduced_hull_mask[idx_previous_hull_point] = False
            if convex_distance(points[idx_previous_hull_point], points[reduced_hull_mask]) <= interior_thresh:
                # Remove this as a hull point.
                n_hull_removed += 1
                hull_mask[idx_previous_hull_point] = False
            else:
                # The point wasn't interior, so add it back in.
                reduced_hull_mask[idx_previous_hull_point] = True

        print("Eliminated {} interior points.".format(len(idxs_interior_points) - 1 + n_hull_removed))

        # If we hit either of our stopping criteria, return now.
        if (
            approximation_error <= desired_approximation_error or
            np.sum(hull_mask) >= max_n_vertices
        ):
            return np.nonzero(hull_mask)[0], approximation_error

# FIXME: remove this.
if __name__ == "__main__":

    dims = np.arange(2, 11)
    times = np.empty_like(dims, dtype=float)
    n_points = 100

    from qinfer.perf_testing import timing

    for idx_dim, dim in enumerate(dims):
        print("======= DIM {} =======".format(dim))
        points = np.random.randn(n_points, dim)
        points /= np.linalg.norm(points, axis=0)
        points *= np.random.random((n_points,))[:, None]
        with timing() as t:
            hull_pts, eps = approximate_convex_hull(points, n_points - 1, 0, seed_method='extreme')
        times[idx_dim] = t.delta_t
        print(t)
