#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_abstract_model.py: Checks that Model works properly.
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
