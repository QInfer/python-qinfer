#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# base_test.py: Base class for derandomized test classes.
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
import unittest

## CLASSES ####################################################################

class DerandomizedTestCase(unittest.TestCase):

    ## SETUP AND TEARDOWN ##
    # We want every test method to be setup first by seeding NumPy's random
    # number generator with a predictable seed (namely: zero). This way,
    # all of our tests are *deterministic*, and once first checked, will
    # not deviate from that behavior unless there is a change to the underlying
    # functionality.
    #
    # We do this by using the fact that nosetests and unittest both call
    # the method named "setUp" (note the capitalization!) before each
    # test method.
    
    def setUp(self):
        # TODO: move this into a base class for deteministic unit tests!
        np.random.seed(0)
        
