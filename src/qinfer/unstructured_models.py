#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# unstructured_models.py: Models for unstructured or minimally-structured
#     estimation.
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

## FEATURES ##################################################################

from __future__ import absolute_import
from __future__ import division # Ensures that a/b is always a float.

## EXPORTS ###################################################################

__all__ = [
	'UnstructuredFrequencyModel'
]

## IMPORTS ###################################################################

import numpy as np

from qinfer.utils import binomial_pdf
from qinfer.abstract_model import Model, DifferentiableModel

## CLASSES ####################################################################

class UnstructuredFrequencyModel(Model):
	r"""
	Represents estimating a likelihood function of the form

	.. math::

		\Pr(0 | \vec{a}, \vec{\omega}, \vec{\phi}; t) = \sum_i a_i \cos^2(\omega_i t + \phi_i),

	where :math:`\vec{x} = (\vec{a}, \vec{\omega}, \vec{\phi})` are the
	model parameters.
	"""

	# TODO: INITIALIZER
	#       Needs to take number of peaks.

	# TODO: DEFINE PARAMETERS
	#       expparams is just a float, but modelparams is 3 * n_peaks
	#       define names, number

	# TODO: DEFINE VALIDITY

	# TODO: CANONICALIZE
	#       this will be a pain: sort by omega, keeping a and phi in same order,
	#       then normalize to sum_i a_i = 1. Is this valid?