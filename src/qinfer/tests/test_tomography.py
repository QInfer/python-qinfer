#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_tomography.py: Tests tomography distributions and models.
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

from __future__ import division # Ensures that a/b is always a float.

## IMPORTS ####################################################################

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from qinfer.tests.base_test import DerandomizedTestCase
from qinfer.tomography import (
    GinibreQubitDistribution, dm_to_mps, mps_to_dm
)
    
## CLASSES ####################################################################

class TestStateTomography(DerandomizedTestCase):
    # TODO
    
    
    ## TEST METHODS ##
    
    def test_ginibre_round_trip_trace(self):
        """
        State tomography: Asserts that round-tripping Ginibre samples preserves Tr.
        """
        dist = GinibreQubitDistribution(3)
        
        samples = dist.sample()

        assert_almost_equal(mps_to_dm(samples)[0].tr(), 1)
