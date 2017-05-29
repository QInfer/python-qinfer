#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_test.py: Checks that utilities for unit testing
#     actually run the tests we expect.
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

from qinfer import UniformDistribution
from qinfer.tests.base_test import (
    DerandomizedTestCase, MockModel, assert_warns, test_model
)

## TESTS #####################################################################

class TestTest(DerandomizedTestCase):

    def test_assert_warns_ok(self):
        with assert_warns(RuntimeWarning):
            warnings.warn(RuntimeWarning("Test"))

    @unittest.expectedFailure
    def test_assert_warns_nowarn(self):
        with assert_warns(RuntimeWarning):
            pass
    
    def test_test_model_runs(self):
        model = MockModel()
        prior = UniformDistribution(np.array([[10,12],[2,3]]))
        eps = np.arange(10,20).astype(model.expparams_dtype)
        test_model(model, prior, eps)
        
