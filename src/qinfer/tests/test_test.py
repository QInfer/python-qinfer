#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_test.py: Checks that utilities for unit testing
#     actually run the tests we expect.
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
        
