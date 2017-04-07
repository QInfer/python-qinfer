#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# domains.py: module for domains of model outcomes
##
# © 2012 Chris Ferrie (csferrie@gmail.com) and
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

## IMPORTS ###################################################################

from __future__ import division
from __future__ import absolute_import

from builtins import range
from future.utils import with_metaclass
from functools import reduce

from operator import mul
from scipy.special import binom
from math import factorial
from itertools import combinations_with_replacement, product
import numpy as np
from .utils import join_struct_arrays, separate_struct_array

import abc

import warnings

## EXPORTS ###################################################################

__all__ = [
    'Domain',
    'ProductDomain',
    'RealDomain',
    'IntegerDomain',
    'MultinomialDomain'
]

## FUNCTIONS #################################################################

## ABSTRACT CLASSES AND MIXINS ###############################################

class Domain(with_metaclass(abc.ABCMeta, object)):
    """
    Abstract base class for domains of outcomes of models.
    """

    ## ABSTRACT PROPERTIES ##

    @abc.abstractproperty
    def is_continuous(self):
        """
        Whether or not the domain has an uncountable number of values.

        :type: `bool`
        """
        pass

    @abc.abstractproperty
    def is_finite(self):
        """
        Whether or not the domain contains a finite number of points.

        :type: `bool`
        """
        pass

    @abc.abstractproperty
    def dtype(self):
        """
        The numpy dtype of a single element of the domain.

        :type: `np.dtype`
        """
        pass

    @abc.abstractproperty
    def n_members(self):
        """
        Returns the number of members in the domain if it
        `is_finite`, otherwise, returns `np.inf`.

        :type: ``int`` or ``np.inf``
        """
        pass

    @abc.abstractproperty
    def example_point(self):
        """
        Returns any single point guaranteed to be in the domain, but
        no other guarantees; useful for testing purposes.
        This is given as a size 1 ``np.array`` of type `dtype`.

        :type: ``np.ndarray``
        """
        pass

    @abc.abstractproperty
    def values(self):
        """
        Returns an `np.array` of type `dtype` containing
        some values from the domain.
        For domains where `is_finite` is ``True``, all elements
        of the domain will be yielded exactly once.

        :rtype: `np.ndarray`
        """
        pass

    ## CONCRETE PROPERTIES ##

    @property
    def is_discrete(self):
        """
        Whether or not the domain has a countable number of values.

        :type: `bool`
        """
        return not self.is_continuous

    ## ABSTRACT METHODS ##

    @abc.abstractmethod
    def in_domain(self, points):
        """
        Returns ``True`` if all of the given points are in the domain,
        ``False`` otherwise.

        :param np.ndarray points: An `np.ndarray` of type `self.dtype`.

        :rtype: `bool`
        """
        pass

class ProductDomain(Domain):
    """
    A domain made from the cartesian product of other domains.

    :param Domain domains: ``Domain`` instances as separate arguments, 
        or as a singe list of ``Domain`` instances.
    """
    def __init__(self, *domains):
        
        if len(domains) == 1:
            try:
                self._domains = list(domains[0])
            except:
                self._domains = domains
        else:
            self._domains = domains
        
        self._domains = domains
        self._dtypes = [domain.example_point.dtype for domain in self._domains]
        self._example_point = join_struct_arrays(
            [np.array(domain.example_point) for domain in self._domains]
        )
        self._dtype = self._example_point.dtype

    @property
    def is_continuous(self):
        """
        Whether or not the domain has an uncountable number of values.

        :type: `bool`
        """
        return any([domain.is_continuous for domain in self._domains])

    @property
    def is_finite(self):
        """
        Whether or not the domain contains a finite number of points.

        :type: `bool`
        """
        return all([domain.is_finite for domain in self._domains])

    @property
    def dtype(self):
        """
        The numpy dtype of a single element of the domain.

        :type: `np.dtype`
        """
        return self._dtype

    @property
    def n_members(self):
        """
        Returns the number of members in the domain if it
        `is_finite`, otherwise, returns `np.inf`.

        :type: ``int`` or ``np.inf``
        """
        if self.is_finite:
            return reduce(mul, [domain.n_members for domain in self._domains], 1)
        else:
            return np.inf

    @property
    def example_point(self):
        """
        Returns any single point guaranteed to be in the domain, but
        no other guarantees; useful for testing purposes.
        This is given as a size 1 ``np.array`` of type `dtype`.

        :type: ``np.ndarray``
        """
        return self._example_point

    @property
    def values(self):
        """
        Returns an `np.array` of type `dtype` containing
        some values from the domain.
        For domains where `is_finite` is ``True``, all elements
        of the domain will be yielded exactly once.

        :rtype: `np.ndarray`
        """
        separate_values = [domain.values for domain in self._domains]
        return np.concatenate([
            join_struct_arrays(list(map(np.array, value))) 
            for value in product(*separate_values)
        ])

    ## METHODS ##
    
    def _mytype(self, array):
        # astype does weird stuff with struct names, and possibly 
        # depends on numpy version; hopefully 
        # the following is a bit more predictable since it passes through 
        # uint8
        return separate_struct_array(array, self.dtype)[0]
    
    def to_regular_arrays(self, array):
        """
        Expands from an array of type `self.dtype` into a list of
        arrays with dtypes corresponding to the factor domains.

        :param np.ndarray array: An `np.array` of type `self.dtype`.

        :rtype: ``list``
        """
        return separate_struct_array(self._mytype(array), self._dtypes)

    def from_regular_arrays(self, arrays):
        """
        Merges a list of arrays (of the same shape) of dtypes 
        corresponding to the factor domains into a single array 
        with the dtype of the ``ProductDomain``.

        :param list array: A list with each element of type ``np.ndarray``

        :rtype: `np.ndarray`
        """
        return self._mytype(join_struct_arrays([
            array.astype(dtype)
            for dtype, array in zip(self._dtypes, arrays)
        ]))
        

    def in_domain(self, points):
        """
        Returns ``True`` if all of the given points are in the domain,
        ``False`` otherwise.

        :param np.ndarray points: An `np.ndarray` of type `self.dtype`.

        :rtype: `bool`
        """
        return all([
            domain.in_domain(array)
            for domain, array in
                zip(self._domains, separate_struct_array(points, self._dtypes))
        ])


## CLASSES ###################################################################

class RealDomain(Domain):
    """
    A domain specifying a contiguous (and possibly open ended) subset
    of the real numbers.

    :param float min: A number specifying the lowest possible value of the
        domain.
    :param float max: A number specifying the largest possible value of the
        domain.
    """

    def __init__(self, min=-np.inf, max=np.inf):
        self._min = min
        self._max = max

    ## PROPERTIES ##

    @property
    def min(self):
        """
        Returns the minimum value of the domain.

        :rtype: `float`
        """
        return self._min
    @property
    def max(self):
        """
        Returns the maximum value of the domain.

        :rtype: `float`
        """
        return self._max

    @property
    def is_continuous(self):
        """
        Whether or not the domain has an uncountable number of values.

        :type: `bool`
        """
        return True

    @property
    def is_finite(self):
        """
        Whether or not the domain contains a finite number of points.

        :type: `bool`
        """
        return False

    @property
    def dtype(self):
        """
        The numpy dtype of a single element of the domain.

        :type: `np.dtype`
        """
        return np.float

    @property
    def n_members(self):
        """
        Returns the number of members in the domain if it
        `is_finite`, otherwise, returns `None`.

        :type: ``np.inf``
        """
        return np.inf

    @property
    def example_point(self):
        """
        Returns any single point guaranteed to be in the domain, but
        no other guarantees; useful for testing purposes.
        This is given as a size 1 ``np.array`` of type ``dtype``.

        :type: ``np.ndarray``
        """
        if not np.isinf(self.min):
            return np.array([self.min], dtype=self.dtype)
        if not np.isinf(self.max):
            return np.array([self.max], dtype=self.dtype)
        else:
            return np.array([0], dtype=self.dtype)

    @property
    def values(self):
        """
        Returns an `np.array` of type `self.dtype` containing
        some values from the domain.
        For domains where ``is_finite`` is ``True``, all elements
        of the domain will be yielded exactly once.

        :rtype: `np.ndarray`
        """
        return self.example_point

    ## METHODS ##

    def in_domain(self, points):
        """
        Returns ``True`` if all of the given points are in the domain,
        ``False`` otherwise.

        :param np.ndarray points: An `np.ndarray` of type `self.dtype`.

        :rtype: `bool`
        """
        if np.all(np.isreal(points)):
            are_greater = np.all(np.greater_equal(points, self._min))
            are_smaller = np.all(np.less_equal(points, self._max))
            return  are_greater and are_smaller
        else:
            return False

class IntegerDomain(Domain):
    """
    A domain specifying a contiguous (and possibly open ended) subset
    of the integers.

    Internally minimum and maximum are represented as
    floats in order to handle the case of infinite maximum, and minimums. The
    integer conversion function will be applied to the min and max values.

    :param int min: A number specifying the lowest possible value of the
        domain.
    :param int max: A number specifying the largest possible value of the
        domain.

    Note: Yes, it is slightly unpythonic to specify `max` instead of `max`+1.
    """

    def __init__(self, min=0, max=np.inf):
        self._min = int(min) if not np.isinf(min) else min
        self._max = int(max) if not np.isinf(max) else max

    ## PROPERTIES ##

    @property
    def min(self):
        """
        Returns the minimum value of the domain.

        :rtype: `float` or `np.inf`
        """
        return int(self._min) if not np.isinf(self._min) else self._min
    @property
    def max(self):
        """
        Returns the maximum value of the domain.

        :rtype: `float` or `np.inf`
        """
        return int(self._max) if not np.isinf(self._max) else self._max


    @property
    def is_continuous(self):
        """
        Whether or not the domain has an uncountable number of values.

        :type: `bool`
        """
        return False

    @property
    def is_finite(self):
        """
        Whether or not the domain contains a finite number of points.

        :type: `bool`
        """
        return not np.isinf(self.min) and not np.isinf(self.max)

    @property
    def dtype(self):
        """
        The numpy dtype of a single element of the domain.

        :type: `np.dtype`
        """
        return np.int

    @property
    def n_members(self):
        """
        Returns the number of members in the domain if it
        `is_finite`, otherwise, returns `np.inf`.

        :type: ``int`` or ``np.inf``
        """
        if self.is_finite:
            return int(self.max - self.min + 1)
        else:
            return np.inf

    @property
    def example_point(self):
        """
        Returns any single point guaranteed to be in the domain, but
        no other guarantees; useful for testing purposes.
        This is given as a size 1 ``np.array`` of type ``dtype``.

        :type: ``np.ndarray``
        """
        if not np.isinf(self.min):
            return np.array([self._min], dtype=self.dtype)
        if not np.isinf(self.max):
            return np.array([self._max], dtype=self.dtype)
        else:
            return np.array([0], dtype=self.dtype)

    @property
    def values(self):
        """
        Returns an `np.array` of type `self.dtype` containing
        some values from the domain.
        For domains where ``is_finite`` is ``True``, all elements
        of the domain will be yielded exactly once.

        :rtype: `np.ndarray`
        """
        if self.is_finite:
            return np.arange(self.min, self.max + 1, dtype = self.dtype)
        else:
            return self.example_point

    ## METHODS ##

    def in_domain(self, points):
        """
        Returns ``True`` if all of the given points are in the domain,
        ``False`` otherwise.

        :param np.ndarray points: An `np.ndarray` of type `self.dtype`.

        :rtype: `bool`
        """
        if np.all(np.isreal(points)):
            try:
                are_integer = np.all(np.mod(points, 1) == 0)
            except TypeError:
                are_integer = False
            are_greater = np.all(np.greater_equal(points, self._min))
            are_smaller = np.all(np.less_equal(points, self._max))
            return  are_integer and are_greater and are_smaller
        else:
            return False

class MultinomialDomain(Domain):
    """
    A domain specifying k-tuples of non-negative integers which
    sum to a specific value.

    :param int n_meas: The sum of any tuple in the domain.
    :param int n_elements: The number of elements in a tuple.
    """

    def __init__(self, n_meas, n_elements=2):
        self._n_elements = n_elements
        self._n_meas = n_meas

    ## PROPERTIES ##

    @property
    def n_meas(self):
        """
        Returns the sum of any tuple in the domain.

        :rtype: `int`
        """
        return self._n_meas
    @property
    def n_elements(self):
        """
        Returns the number of elements of a tuple in the domain.

        :rtype: `int`
        """
        return self._n_elements


    @property
    def is_continuous(self):
        """
        Whether or not the domain has an uncountable number of values.

        :type: `bool`
        """
        return False

    @property
    def is_finite(self):
        """
        Whether or not the domain contains a finite number of points.

        :type: `bool`
        """
        return True

    @property
    def dtype(self):
        """
        The numpy dtype of a single element of the domain.

        :type: `np.dtype`
        """
        return np.dtype([('k', np.int, self.n_elements)])

    @property
    def n_members(self):
        """
        Returns the number of members in the domain if it
        `is_finite`, otherwise, returns `None`.

        :type: ``int``
        """
        return int(binom(self.n_meas + self.n_elements -1, self.n_elements - 1))

    @property
    def example_point(self):
        """
        Returns any single point guaranteed to be in the domain, but
        no other guarantees; useful for testing purposes.
        This is given as a size 1 ``np.array`` of type ``dtype``.

        :type: ``np.ndarray``
        """
        return np.array([([self.n_meas] + [0] * (self.n_elements-1),)], dtype=self.dtype)

    @property
    def values(self):
        """
        Returns an `np.array` of type `self.dtype` containing
        some values from the domain.
        For domains where ``is_finite`` is ``True``, all elements
        of the domain will be yielded exactly once.

        :rtype: `np.ndarray`
        """

        # This code comes from Jared Goguen at http://stackoverflow.com/a/37712597/1082565
        partition_array = np.empty((self.n_members, self.n_elements), dtype=int)
        masks = np.identity(self.n_elements, dtype=int)
        for i, c in enumerate(combinations_with_replacement(masks, self.n_meas)):
            partition_array[i,:] = sum(c)

        # Convert to dtype before returning
        return self.from_regular_array(partition_array)

    ## METHODS ##

    def to_regular_array(self, A):
        """
        Converts from an array of type `self.dtype` to an array
        of type `int` with an additional index labeling the
        tuple indeces.

        :param np.ndarray A: An `np.array` of type `self.dtype`.

        :rtype: `np.ndarray`
        """
        # this could be a static method, but we choose to be consistent with
        # from_regular_array
        return A.view((int, len(A.dtype.names))).reshape(A.shape + (-1,))

    def from_regular_array(self, A):
        """
        Converts from an array of type `int` where the last index
        is assumed to have length `self.n_elements` to an array
        of type `self.d_type` with one fewer index.

        :param np.ndarray A: An `np.array` of type `int`.

        :rtype: `np.ndarray`
        """
        dims = A.shape[:-1]
        return A.reshape((np.prod(dims),-1)).view(dtype=self.dtype).squeeze(-1).reshape(dims)

    def in_domain(self, points):
        """
        Returns ``True`` if all of the given points are in the domain,
        ``False`` otherwise.

        :param np.ndarray points: An `np.ndarray` of type `self.dtype`.

        :rtype: `bool`
        """
        array_view = self.to_regular_array(points)
        non_negative = np.all(np.greater_equal(array_view, 0))
        correct_sum = np.all(np.sum(array_view, axis=-1) == self.n_meas)
        return non_negative and correct_sum
