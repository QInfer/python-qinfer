#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_metrics.py: Tests various metrics like risk and information gain.
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

from __future__ import division # Ensures that a/b is always a float.
from __future__ import absolute_import
## IMPORTS ####################################################################

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_array_less,assert_approx_equal

from qinfer.tests.base_test import DerandomizedTestCase
from qinfer import (BinomialModel,CoinModel,BetaDistribution,DifferentiableBinomialModel)

from qinfer.smc import SMCUpdater,SMCUpdaterBCRB

class TestBayesRisk(DerandomizedTestCase):
    # Test the implementation of numerical Bayes Risk by comparing to 
    # numbers which were derived by doing analytic/numeric
    # integrals of simple models in Mathematica. This test trusts that 
    # these calculations were done correctly.

    ALPHA = 1.
    BETA = 3.
    PRIOR_BETA = BetaDistribution(alpha=ALPHA, beta=BETA)
    N_PARTICLES = 10000
    NMEAS_EXPPARAMS = np.arange(1, 11, dtype=int)
    
    def setUp(self):

        super(TestBayesRisk,self).setUp()
        
        # Set up relevant models.
        self.coin_model = CoinModel()
        self.binomial_model = BinomialModel(self.coin_model)

        # Set up updaters for these models using particle approximations 
        # of conjugate priors
        self.updater_binomial = SMCUpdater(self.binomial_model,
                TestBayesRisk.N_PARTICLES,TestBayesRisk.PRIOR_BETA)

    def test_finite_outcomes_risk(self):
        # The binomial model has a finite number of outcomes. Test the 
        # risk calculation in this case.

        expparams = self.NMEAS_EXPPARAMS.astype(self.binomial_model.expparams_dtype)

        # estimate the risk
        est_risk = self.updater_binomial.bayes_risk(expparams)

        # compute exact risk
        a, b = TestBayesRisk.ALPHA, TestBayesRisk.BETA
        exact_risk = a * b / ((a + b) * (a + b + 1) * (a + b + expparams['n_meas']))

        # see if they roughly match
        assert_almost_equal(est_risk, exact_risk, decimal=3)

class TestInformationGain(DerandomizedTestCase):
    # Test the implementation of numerical information gain by comparing to 
    # numbers which were derived by doing analytic/numeric
    # integrals of simple models (binomialm, poisson, and gaussian) in 
    # Mathematica. This test trusts that these calculations
    # were done correctly.

    ALPHA = 1
    BETA = 3
    PRIOR_BETA = BetaDistribution(alpha=ALPHA, beta=BETA)
    N_PARTICLES = 10000
    # Calculated in Mathematica, IG for the binomial model and the given expparams
    NMEAS_EXPPARAMS = np.arange(1, 11, dtype=int)
    BINOM_IG = np.array([0.104002,0.189223,0.261496,0.324283,0.379815,0.429613,0.474764,0.516069,0.554138,0.589446])
    
    def setUp(self):

        super(TestInformationGain,self).setUp()
        
        # Set up relevant models.
        self.coin_model = CoinModel()
        self.binomial_model = BinomialModel(self.coin_model)
        
        # Set up updaters for these models using particle approximations 
        # of conjugate priors
        self.updater_binomial = SMCUpdater(self.binomial_model,
                TestInformationGain.N_PARTICLES,TestInformationGain.PRIOR_BETA)


    def test_finite_outcomes_ig(self):
        # The binomial model has a finite number of outcomes. Test the 
        # ig calculation in this case.

        expparams = self.NMEAS_EXPPARAMS.astype(self.binomial_model.expparams_dtype)

        # estimate the information gain
        est_ig = self.updater_binomial.expected_information_gain(expparams)

        # see if they roughly match
        assert_almost_equal(est_ig, TestInformationGain.BINOM_IG, decimal=2)

class TestFisherInformation(DerandomizedTestCase):
    # Test the implementation of numerical Fisher Information by comparing to 
    # numbers which were derived by doing analytic/numeric
    # integrals of simple models (binomialm, poisson, and gaussian) in 
    # Mathematica. This test trusts that these calculations
    # were done correctly.

    ALPHA = 1
    BETA = 3
    PRIOR_BETA = BetaDistribution(alpha=ALPHA, beta=BETA)
    N_PARTICLES = 10000

    BIN_FI_MODELPARAMS = np.linspace(0.01,0.99,5)
    NMEAS_EXPPARAMS = np.arange(1, 11, dtype=int)
    
    def setUp(self):

        super(TestFisherInformation,self).setUp()
        
        # Set up relevant models.
        self.coin_model = CoinModel()
        self.binomial_model = DifferentiableBinomialModel(self.coin_model)

        # Set up updaters for these models using particle approximations 
        # of conjugate priors
        self.updater_binomial = SMCUpdater(self.binomial_model,
                TestFisherInformation.N_PARTICLES,TestFisherInformation.PRIOR_BETA)


    def test_finite_outcomes_fi(self):
        # The binomial model has a finite number of outcomes. Test the 
        # ig calculation in this case.

        expparams = self.NMEAS_EXPPARAMS.astype(self.binomial_model.expparams_dtype)
        p = TestFisherInformation.BIN_FI_MODELPARAMS
        # estimate the information gain
        est_fi = self.binomial_model.fisher_information(TestFisherInformation.BIN_FI_MODELPARAMS,expparams)[0,0]
        p = p[:,np.newaxis]
        n = expparams.astype(np.float32)[np.newaxis,:]

        exact_fi = n/((1-p)*p) 
        # see if they roughly match
        
        assert_almost_equal(est_fi,exact_fi, decimal=3)
