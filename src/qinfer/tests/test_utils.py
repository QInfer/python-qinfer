#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_utils.py: Tests helper functions in utils.py
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

import warnings
import unittest

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from qinfer.tests.base_test import DerandomizedTestCase, MockModel, assert_warns

from qinfer.utils import in_ellipsoid, assert_sigfigs_equal

## TESTS #####################################################################

class TestNumericTests(DerandomizedTestCase):

    def test_assert_sigfigs_equal(self):
        """
        Tests to make sure assert_sigfigs_equal
        only passes if the correct number of 
        significant figures match
        """

        # these are the same to 6 sigfigs
        assert_sigfigs_equal(
            np.array([3.141592]),
            np.array([3.141593]),
            6
        )
        # these are only the same to 5 sigfigs
        self.assertRaises(
            AssertionError,
            assert_sigfigs_equal,
            np.array([3.14159]),
            np.array([3.14158]),
            6
        )

        # these are the same to 3 sigfigs
        assert_sigfigs_equal(
            np.array([1729]),
            np.array([1728]),
            3
        )
        # these are only the same to 3 sigfigs
        self.assertRaises(
            AssertionError,
            assert_sigfigs_equal,
            np.array([1729]),
            np.array([1728]),
            4
        )
        

class TestEllipsoids(DerandomizedTestCase):

    def test_in_ellipsoid(self):
        
        # the semi-major axes are the square roots of the 
        # singular values, so 2 and 1 in this case.
        A = np.array([[4,0], [0,1]])
        c = np.array([0,1])

        # test with multiple inputs. account for numerical error at boundary.
        x = np.array([[10,5],[0,1],[0,2],[0,3],[2,1],[3,1],[0.5,1.5]])
        assert_equal(
            in_ellipsoid(x, A, c), 
            np.array([0, 1, 1, 0, 1, 0, 1],dtype=bool)
        )
        # test with single input
        assert(in_ellipsoid(c,A,c))

        # Random positive matrix and origin
        A = np.random.randn(5, 5)
        A = np.dot(A, A.T)
        c = np.random.randn(5)

        # Look along a couple of the semi-major axes
        U, s, _ = np.linalg.svd(A)
        x = np.vstack([
            c + 0.99 * np.sqrt(s[2]) * U[:,2],
            c + 1.01 * np.sqrt(s[2]) * U[:,2],
            c - 0.99 * np.sqrt(s[0]) * U[:,0],
            c - 1.01 * np.sqrt(s[0]) * U[:,0],
        ])
        assert_equal(
            in_ellipsoid(x, A, c), 
            np.array([1,0,1,0], dtype=bool)
        )