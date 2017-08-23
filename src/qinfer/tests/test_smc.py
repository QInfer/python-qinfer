#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_smc.py: Checks that properties and methods of
#     SMCUpdater work as intended.
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

from qinfer.tests.base_test import DerandomizedTestCase, MockModel, assert_warns
from qinfer.abstract_model import FiniteOutcomeModel
from qinfer.distributions import UniformDistribution
from qinfer.smc import SMCUpdater
from qinfer._exceptions import ApproximationWarning
    
## CLASSES ####################################################################

class DecimationModel(MockModel):
    r"""
    Two-outcome model whose likelihood upon a "0" outcome returns 1 for the
    first :math:`\alpha` of its model parameters and
    0 for the rest, where :math:`\alpha` is an experiment parameter.
    As with most other mock models, this is not a valid statistical
    model, but is 
    useful in decimating particle clouds in tests,
    as this reduces the ESS of an SMC updater using this
    model by a factor of :math:`\alpha` after each "datum."
    """
        
    @property
    def expparams_dtype(self):
        return [('alpha', float)]        
    
    def likelihood(self, outcomes, modelparams, expparams):
        super(DecimationModel, self).likelihood(outcomes, modelparams, expparams)

        assert expparams.shape == (1,) # Only defined for single experiments.

        pr0 = np.ones((modelparams.shape[0], expparams.shape[0])) / 2
        idx_dec_at = np.ceil(expparams['alpha'][0] * modelparams.shape[0]).astype(np.int)
        pr0[idx_dec_at:, :] = 0 
        
        return FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)

## TEST CASES ################################################################

class TestSMCEffectiveSampleSize(DerandomizedTestCase):
    """
    Tests that the SMCUpdater class correctly implements the effective
    sample size criterion as a resampling threshold, postselection
    herald, etc.
    """
    
    def setUp(self):
        super(TestSMCEffectiveSampleSize, self).setUp()
        self.model = DecimationModel()

    def _mk_updater(self, n_particles, **kwargs):
        return SMCUpdater(self.model, n_particles, UniformDistribution([0, 1]), **kwargs)

    def test_low_n_ess_warning(self):
        n_particles = 1000
        updater = self._mk_updater(n_particles, resample_thresh=0.0)
        
        outcomes = np.array([0], dtype=int)
        expparams = np.ones((1,), dtype=self.model.expparams_dtype)
        expparams['alpha'][0] = 2 / 1000 # Force the particle number to be 2.

        with assert_warns(ApproximationWarning):
            updater.update(outcomes, expparams) 

    def test_resample_thresh(self):
        n_updates = 10
        n_particles = 1000
        updater = self._mk_updater(n_particles, resample_thresh=0.5)

        outcomes = np.array([0], dtype=int)
        expparams = np.ones((1,), dtype=self.model.expparams_dtype)
        expparams['alpha'][0] = 0.3 # Something less than the threshold, force resampling.

        for idx_update in range(n_updates):
            updater.update(outcomes, expparams)
            assert_equal(updater.resample_count, 1 + idx_update)

    def test_min_n_ess(self):
        n_updates = 6
        n_particles = 4 ** n_updates # Pick factor of 4 to avoid discretization errors.
        updater = self._mk_updater(n_particles, resample_thresh=0.0)

        outcomes = np.array([0], dtype=int)
        expparams = np.empty((1,), dtype=self.model.expparams_dtype)

        for idx_update in range(n_updates):
            expparams['alpha'][0] = 4 ** -(idx_update + 1)
            updater.update(outcomes, expparams)
            assert_equal(updater.min_n_ess, 4 ** (n_updates - idx_update - 1))

        # Force a resample and ensure that the min_n_ess remains the same.
        updater.resample()
        assert_equal(updater.min_n_ess, 4 ** (n_updates - idx_update - 1))
