#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_abstract_model.py: Checks that Model works properly.
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
from numpy.testing import assert_equal, assert_almost_equal

from qinfer.tests.base_test import DerandomizedTestCase, MockModel
from qinfer.abstract_model import FiniteOutcomeModel
    
## CLASSES ####################################################################

class TestAbstractModel(DerandomizedTestCase):
    
    def setUp(self):
        super(TestAbstractModel, self).setUp()
        self.mock_model = MockModel()

    def test_pr0_shape(self):
        """
        Model: Checks that pr0-based Model subtypes give the right shape.
        """
        outcomes = np.array([0, 1], dtype=int)
        modelparams = np.random.random((3, 2))
        expparams = np.zeros((4,), dtype=self.mock_model.expparams_dtype)
        
        assert self.mock_model.likelihood(outcomes, modelparams, expparams).shape == (2, 3, 4)
        
    def test_simulate_experiment(self):
        """
        Model: Checks that simulate_experiment behaves correctly.
        """
        modelparams = np.random.random((1, 2))
        expparams = np.zeros((1,), dtype=self.mock_model.expparams_dtype)
        
        assert_almost_equal(
            self.mock_model.simulate_experiment(modelparams, expparams, repeat=2000).mean(),
            0.5,
            1
        )
