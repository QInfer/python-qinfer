#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_domains.py: Checks that built-in instances of Domain work properly.
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
from future.utils import with_metaclass

## IMPORTS ####################################################################

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from qinfer.tests.base_test import (
    DerandomizedTestCase,
    ConcreteDomainTest
)
import abc
from qinfer import (
    Domain, RealDomain, IntegerDomain, MultinomialDomain
)

import unittest

## SIMPLE TEST MODELS #########################################################

class TestIntegerDomain(ConcreteDomainTest, DerandomizedTestCase):
    """
    Tests IntegerDomain with all integers.
    """

    def instantiate_domain(self):
        return IntegerDomain()
    def instantiate_good_values(self):
        [np.array([54]).astype(np.float), np.array([-32]).astype(np.int)]
    def instantiate_bad_values(self):
        [np.array([0.5]), np.array([1j])]
