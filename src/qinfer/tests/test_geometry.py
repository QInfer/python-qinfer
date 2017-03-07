#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_geometry.py: Checks correctness of computational geometry functionality.
##
# © 2014 Chris Ferrie (csferrie@gmail.com) and
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
from __future__ import division # Ensures that a/b is always a float.

## IMPORTS ####################################################################

import numpy as np
from numpy.testing import assert_almost_equal

from qinfer.tests.base_test import DerandomizedTestCase

from qinfer.geometry import convex_distance
from qinfer.utils import (
    assert_sigfigs_equal, requires_optional_module,
)
from qinfer.perf_testing import numpy_err_policy

## TESTS #####################################################################

class TestConvexGeometry(DerandomizedTestCase):

    @requires_optional_module("cvxopt", if_missing="skip")
    def test_convex_distance_known_cases(self, dimension=3):
        """
        Convex geometry: Checks convex_distance for known cases.
        """
        assert_almost_equal(
            convex_distance(np.ones((dimension,)), np.eye(dimension)),
            np.linalg.norm(((dimension - 1) / dimension) * np.ones(dimension,), ord=2)
        )

        # TODO: add cases.

    @requires_optional_module("cvxopt", if_missing="skip")
    def test_convex_distance_interior_point(self, n_points=500, dimension=5):
        """
        Convex geometry: Checks that convex_distance is zero for interior pts.
        """
        S = np.random.random((n_points, dimension))
        alpha = np.random.random((n_points))
        alpha /= alpha.sum()

        z = np.dot(alpha, S)

        with numpy_err_policy(invalid='raise'):
            assert_almost_equal(
                convex_distance(z, S),
                0
            )
        
