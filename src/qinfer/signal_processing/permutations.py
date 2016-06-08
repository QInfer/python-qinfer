#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# permutations.py: Permutations for signal processing.
##
# © 2015 Chris Ferrie (csferrie@gmail.com) and
#        Christopher E. Granade (cgranade@gmail.com).
#
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

## FEATURES ##################################################################

from __future__ import division

## IMPORTS ###################################################################

import numpy as np

## FUNCTIONS #################################################################

def bitreverseaxis(arr, axis=0):
    return ditreverseaxis(arr, axis=axis, base=2)

def ditreverseaxis(arr, axis=0, base=2):
    ndits = np.log(arr.shape[axis]) / np.log(base)
    if np.abs(ndits - int(ndits)) > 1e-10:
        raise ValueError("Expected a power of {}, got {}.".format(base, arr.shape[axis]))
    ndits = int(ndits)

    orig_shape = arr.shape
    shape_left = orig_shape[:axis]
    shape_right = orig_shape[axis+1:]
    dim_labels = range(len(shape_left) + len(shape_right) + ndits)
    dim_labels[axis:axis+ndits] = dim_labels[axis:axis+ndits][::-1]
    arr = arr.reshape(shape_left + (base,) * ndits + shape_right).transpose(dim_labels).reshape(orig_shape)

    return arr



## TESTS #####################################################################
# TODO
