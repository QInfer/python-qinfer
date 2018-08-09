#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# unstructured_models.py: Models for unstructured or minimally-structured
#     estimation.
##
# © 2017, Chris Ferrie (csferrie@gmail.com) and
#         Christopher Granade (cgranade@cgranade.com).
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the copyright holder nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
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