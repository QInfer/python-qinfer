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
    Domain, ProductDomain,
    RealDomain, IntegerDomain, MultinomialDomain
)

import unittest

## CONSTANTS ###################################################################

WEIRDO = np.array([(1,2.,'jump')], dtype=[('foo', 'i4'),('bar', 'f4'), ('baz', 'S10')])

## DOMAIN TESTS ################################################################

class TestRealDomain(ConcreteDomainTest, DerandomizedTestCase):
    """
    Tests RealDomain with all reals.
    """

    def instantiate_domain(self):
        return RealDomain(min=-np.inf,max=np.inf)
    def instantiate_good_values(self):
        return [np.pi, np.array([2.1]).astype(np.int), np.array([-32.2,2,2.8])]
    def instantiate_bad_values(self):
        return [np.array([1j]), WEIRDO]

class TestPositiveRealDomain(ConcreteDomainTest, DerandomizedTestCase):
    """
    Tests RealDomain with all non-negative reals.
    """

    def instantiate_domain(self):
        return RealDomain(min=0,max=np.inf)
    def instantiate_good_values(self):
        return [0, np.pi, np.array([2.1]).astype(np.int), np.array([32.2,2,2.8])]
    def instantiate_bad_values(self):
        return [-0.001, np.array([1j]), WEIRDO]

class TestNegativeRealDomain(ConcreteDomainTest, DerandomizedTestCase):
    """
    Tests RealDomain with all non-positive reals.
    """

    def instantiate_domain(self):
        return RealDomain(min=-np.inf,max=0)
    def instantiate_good_values(self):
        return [0, -np.pi, np.array([-2.1]).astype(np.int), np.array([-32.2,-2,-2.8])]
    def instantiate_bad_values(self):
        return [0.001, np.array([1j]), WEIRDO]

class TestBoundedRealDomain(ConcreteDomainTest, DerandomizedTestCase):
    """
    Tests RealDomain with a closed interval.
    """

    def instantiate_domain(self):
        return RealDomain(min=np.e,max=np.pi)
    def instantiate_good_values(self):
        return [np.pi, np.array([3]), np.array([np.e,3.13,2.8])]
    def instantiate_bad_values(self):
        return [3.15, np.array([1j]), WEIRDO]

class TestIntegerDomain(ConcreteDomainTest, DerandomizedTestCase):
    """
    Tests IntegerDomain with all integers.
    """

    def instantiate_domain(self):
        return IntegerDomain(min=-np.inf,max=np.inf)
    def instantiate_good_values(self):
        return [np.array([54]).astype(np.float), np.array([-32,2,2]).astype(np.int)]
    def instantiate_bad_values(self):
        return [np.array([0.5]), np.array([np.pi, 1]), 1j, WEIRDO]

class TestPositiveIntegerDomain(ConcreteDomainTest, DerandomizedTestCase):
    """
    Tests IntegerDomain with all non-negative integers.
    """

    def instantiate_domain(self):
        return IntegerDomain(min=0,max=np.inf)
    def instantiate_good_values(self):
        return [3, np.array([54]).astype(np.float), np.array([0, 32,1,2]).astype(np.int)]
    def instantiate_bad_values(self):
        return [np.array([0.5]), np.array([-1, 1]), 1j, WEIRDO]

class TestNegativeIntegerDomain(ConcreteDomainTest, DerandomizedTestCase):
    """
    Tests IntegerDomain with all non-positive integers.
    """

    def instantiate_domain(self):
        return IntegerDomain(min=-np.inf,max=0)
    def instantiate_good_values(self):
        return [-3, np.array([-54]).astype(np.float), np.array([0, -32,-1,-2]).astype(np.int)]
    def instantiate_bad_values(self):
        return [np.array([-0.5]), np.array([-1, 1]), 1j, WEIRDO]

class TestFiniteIntegerDomain(ConcreteDomainTest, DerandomizedTestCase):
    """
    Tests IntegerDomain with a finite integer range.
    """

    def instantiate_domain(self):
        return IntegerDomain(min=-2,max=8)
    def instantiate_good_values(self):
        return [3, np.array([8]).astype(np.float), np.array([0, -2,1,2]).astype(np.int)]
    def instantiate_bad_values(self):
        return [np.array([0.5]), np.array([-5, 1]), 1j, WEIRDO]

class TestMultinomialDomain(ConcreteDomainTest, DerandomizedTestCase):
    """
    Tests MultinomialDomain.
    """

    def instantiate_domain(self):
        return MultinomialDomain(5, n_elements=3)
    def instantiate_good_values(self):
        return [
            np.array([([4,0,1],),], dtype=np.dtype([('l', np.int, 3)])),
            np.array([([1,1,3],),([2,2,1],)], dtype=self.domain.dtype),
            self.domain.from_regular_array(np.array([[1,0,4],[2,3,0]]))
        ]
    def instantiate_bad_values(self):
        return [
            np.array([([4,1,0,1],),], dtype=np.dtype([('l', np.int, 4)])),
            np.array([([1,10,3],),([2,2,1],)], dtype=self.domain.dtype),
            self.domain.from_regular_array(np.array([[-1,0,6],[2,3,0]]))
        ]

    def test_array_conversion(self):
        arr1 = np.array([([1,1,3],)], dtype=self.domain.dtype)
        arr2 = np.array([[1,1,3]])

        assert_equal(self.domain.to_regular_array(arr1), arr2)
        assert_equal(self.domain.from_regular_array(arr2), arr1)
        assert_equal(self.domain.to_regular_array(self.domain.from_regular_array(arr2)), arr2)
        assert_equal(self.domain.from_regular_array(self.domain.to_regular_array(arr1)), arr1)
        
        
class TestIntegerIntegerProductDomain(ConcreteDomainTest, DerandomizedTestCase):
    """
    Tests ProductDomain([IntegerDomain, IntegerDomain])
    """
    
    def instantiate_domain(self):
        return ProductDomain(
            IntegerDomain(min=0,max=5), 
            IntegerDomain(min=-2, max=2)
        )
    def instantiate_good_values(self):
        return [
            np.array([(0,0)], dtype=[('','<i8'),('','<i8')]),
            np.array([(5,2),(0,-2)], dtype=[('','<i8'),('','<i8')])
        ]
    def instantiate_bad_values(self):
        return [
            np.array([(0.5,0)], dtype=[('','f8'),('','<i8')]),
            np.array([(6,2),(0,-2)], dtype=[('','<i8'),('','<i8')])
        ]

    def test_array_conversion(self):
        arr1 = np.array([(5,2),(0,-1)], dtype=[('','<i8'),('','<i8')])
        arr2 = [np.array([5,0]), np.array([2,-1])]

        assert_equal(self.domain.to_regular_arrays(arr1), arr2)
        assert_equal(self.domain.from_regular_arrays(arr2), arr1)
        assert_equal(self.domain.to_regular_arrays(self.domain.from_regular_arrays(arr2)), arr2)
        assert_equal(self.domain.from_regular_arrays(self.domain.to_regular_arrays(arr1)), arr1)

    #override
    def test_is_finite(self):
        assert(self.domain.is_finite)
        assert(self.domain.is_discrete)

class TestIntegerIntegerProductDomain2(ConcreteDomainTest, DerandomizedTestCase):
    """
    Tests ProductDomain([IntegerDomain, IntegerDomain])
    """
    
    def instantiate_domain(self):
        return ProductDomain(
            IntegerDomain(min=0,max=5), 
            IntegerDomain(min=-2, max=np.inf),
            IntegerDomain(min=-np.inf, max=np.inf)
        )
    def instantiate_good_values(self):
        return [
            np.array([(0,0,0)], dtype=[('','<i8'),('','<i8'),('','<i8')]),
            np.array([(5,2,0),(0,-2,10)], dtype=[('','<i8'),('','<i8'),('','<i8')])
        ]
    def instantiate_bad_values(self):
        return [
            np.array([(0.5,0,10)], dtype=[('','f8'),('','<i8'),('','<i8')]),
            np.array([(6,2,10),(0,-2,10)], dtype=[('','<i8'),('','<i8'),('','<i8')])
        ]
        
    #override
    def test_is_finite(self):
        assert(not self.domain.is_finite)
        assert(self.domain.is_discrete)
    
class TestIntegerMultinomialProductDomain(ConcreteDomainTest, DerandomizedTestCase):
    """
    Tests ProductDomain([IntegerDomain, IntegerDomain])
    """
    
    def instantiate_domain(self):
        return ProductDomain(
            IntegerDomain(min=0,max=5), 
            MultinomialDomain(5, n_elements=3)
        )
    def instantiate_good_values(self):
        return [
            np.array([(0,[1,2,2])], dtype=[('','<i8'),('','<i8', 3)]),
            np.array([(5,[5,0,0]),(0,[1,0,4])], dtype=[('','<i8'),('','<i8', 3)])
        ]
    def instantiate_bad_values(self):
        return [
            np.array([(-10,[1,2,2])], dtype=[('','<i8'),('k','<i8', 3)]),
            np.array([(5,[-1,6,0]),(0,[1,0,4])], dtype=[('','<i8'),('k','<i8', 3)])
        ]

    def test_array_conversion(self):
        arr1 = np.array([(5,[1,2,2]),(0,[5,0,0])], dtype=[('','<i8'),('k','<i8', 3)])
        arr2 = [np.array([5,0]), np.array([([1,2,2],), ([5,0,0],)], dtype=[('k','<i8',3)])]

        assert_equal(self.domain.to_regular_arrays(arr1), arr2)
        assert_equal(self.domain.from_regular_arrays(arr2), arr1)
        assert_equal(self.domain.to_regular_arrays(self.domain.from_regular_arrays(arr2)), arr2)
        assert_equal(self.domain.from_regular_arrays(self.domain.to_regular_arrays(arr1)), arr1)
    
