#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_concrete_models.py: Checks that built-in instances of Model work properly.
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
import numpy.lib.recfunctions as rfn

from qinfer.tests.base_test import (
    DerandomizedTestCase, 
    ConcreteDifferentiableModelTest,
    ConcreteModelTest,
    ConcreteSimulatableTest
)
import abc
from qinfer import (
    SimplePrecessionModel, SimpleInversionModel,
    CoinModel, NoisyCoinModel, NDieModel,
    PoisonedModel, BinomialModel,
    MLEModel, RandomWalkModel,
    NormalDistribution,
    BetaDistribution, UniformDistribution,
    ConstrainedSumDistribution
)

import unittest



## TEST MODELS ################################################################

class TestSimplePrecessionModel(ConcreteDifferentiableModelTest, DerandomizedTestCase):
    """
    Tests SimplePrecessionModel.
    """

    def instantiate_model(self):
        return SimplePrecessionModel()
    def instantiate_prior(self):
        return UniformDistribution(np.array([[10,12]]))
    def instantiate_expparams(self):
        return np.arange(10,20).astype(self.model.expparams_dtype)

class TestSimpleInversionModel(ConcreteDifferentiableModelTest, DerandomizedTestCase):
    """
    Tests SimpleInversionModel.
    """

    def instantiate_model(self):
        return SimpleInversionModel()
    def instantiate_prior(self):
        return UniformDistribution(np.array([[5,8]]))
    def instantiate_expparams(self):
        ws = np.linspace(0,0.5,10, dtype=[('w_','float')])
        ts = np.linspace(0,5,10, dtype=[('t','float')])
        return rfn.merge_arrays([ts, ws])

class TestCoinModel(ConcreteDifferentiableModelTest, DerandomizedTestCase):
    """
    Tests CoinModel.
    """

    def instantiate_model(self):
        return CoinModel()
    def instantiate_prior(self):
        return BetaDistribution(mean=0.5, var=0.1)
    def instantiate_expparams(self):
        # only the length of this array matters since CoinModel has no expparams.
        return np.ones((10,),dtype=int)

class TestNoisyCoinModel(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests NoisyCoinModel.
    """

    def instantiate_model(self):
        return NoisyCoinModel()
    def instantiate_prior(self):
        return BetaDistribution(mean=0.5, var=0.1)
    def instantiate_expparams(self):
        alphas = (0.1 * np.ones((10,))).astype([('alpha','float')])
        betas = np.linspace(0,0.5,10, dtype=[('beta','float')])
        return rfn.merge_arrays([alphas,betas])

class TestNDieModel(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests NoisyCoinModel.
    """

    def instantiate_model(self):
        return NDieModel(n=6)
    def instantiate_prior(self):
        unif = UniformDistribution(np.array([[0,1],[0,1],[0,1],[0,1],[0,1],[0,1]]))
        return ConstrainedSumDistribution(unif, desired_total=1)
    def instantiate_expparams(self):
        return np.arange(10).astype(self.model.expparams_dtype)

## DERIVED MODELS #############################################################

class TestBinomialModel(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests BinomialModel with CoinModel as the underlying model
    (underlying model has no expparams).
    """

    def instantiate_model(self):
        return BinomialModel(CoinModel())
    def instantiate_prior(self):
        return BetaDistribution(mean=0.5, var=0.1)
    def instantiate_expparams(self):
        return np.arange(100, 120).astype(self.model.expparams_dtype)

class TestBinomialModel1(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests BinomialModel with SimplePrecessionModel as the underlying model
    (underlying model has 1 scalar expparams).
    """

    def instantiate_model(self):
        return BinomialModel(SimplePrecessionModel())
    def instantiate_prior(self):
        return UniformDistribution(np.array([[5,8]]))
    def instantiate_expparams(self):
        # the scalar expparam is given name 'x' by BinomialModel
        ts = np.linspace(0,5,10, dtype=[('x','float')])
        nmeas = np.arange(10,20).astype([('n_meas','int')])
        return rfn.merge_arrays([ts,nmeas])

class TestBinomialModel2(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests BinomialModel with NoisyCoinModel as the underlying model
    (underlying model has 2 expparams).
    """

    def instantiate_model(self):
        return BinomialModel(NoisyCoinModel())
    def instantiate_prior(self):
        return BetaDistribution(mean=0.5, var=0.1)
    def instantiate_expparams(self):
        alphas = (0.1 * np.ones((10,))).astype([('alpha','float')])
        betas = np.linspace(0,0.5,10, dtype=[('beta','float')])
        nmeas = np.arange(10,20).astype([('n_meas','int')])
        return rfn.merge_arrays([alphas,betas,nmeas])

class TestPoisonedModelALE(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests PoisonedModel with SimplePrecessionModel as the underlying model
    in ALE mode.
    """

    def instantiate_model(self):
        return PoisonedModel(
                SimplePrecessionModel(),
                tol = 1e-4
            )
    def instantiate_prior(self):
        return UniformDistribution(np.array([[5,8]]))
    def instantiate_expparams(self):
        return np.arange(10,20).astype(self.model.expparams_dtype)

class TestPoisonedModelMLE(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests PoisonedModel with SimplePrecessionModel as the underlying model
    in ALE mode.
    """

    def instantiate_model(self):
        return PoisonedModel(
                SimplePrecessionModel(),
                n_samples = 10,
                hedge = 0.01
            )
    def instantiate_prior(self):
        return UniformDistribution(np.array([[5,8]]))
    def instantiate_expparams(self):
        return np.arange(10,20).astype(self.model.expparams_dtype)

class TestMLEModel(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests MLEModel with SimplePrecessionModel and 
    a normal distribution at each step.
    """

    def instantiate_model(self):
        return MLEModel(
                SimplePrecessionModel(),
                likelihood_power = 2
            )
    def instantiate_prior(self):
        return UniformDistribution(np.array([[5,8]]))
    def instantiate_expparams(self):
       return np.arange(10,20).astype(self.model.expparams_dtype)

class TestRandomWalkModel(ConcreteModelTest, DerandomizedTestCase):
    """
    Tests RandomWalkModel with SimplePrecessionModel and 
    a normal distribution at each step.
    """

    def instantiate_model(self):
        return RandomWalkModel(
                SimplePrecessionModel(),
                step_distribution = NormalDistribution(mean=0.1,var=0.1)
            )
    def instantiate_prior(self):
        return UniformDistribution(np.array([[5,8]]))
    def instantiate_expparams(self):
        return np.arange(10,20).astype(self.model.expparams_dtype)