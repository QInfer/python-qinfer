#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_distributions.py: Checks that distribution objects act as expected.
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

## IMPORTS ####################################################################

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from qinfer.tests.base_test import DerandomizedTestCase
from qinfer.distributions import (
    MixtureDistribution,
    NormalDistribution, MultivariateNormalDistribution,
    UniformDistribution, ConstantDistribution, ProductDistribution,
    BetaDistribution, BetaBinomialDistribution, GammaDistribution
)

## CLASSES ####################################################################

class TestNormalDistributions(DerandomizedTestCase):
    """
    Tests ``NormalDistribution`` and ``MultivariateNormalDistribution``
    """

    ## TEST METHODS ##

    def test_normal_moments(self):
        """
        Distributions: Checks that the normal distribtion has the right moments.
        """
        dist = NormalDistribution(0, 1)

        samples = dist.sample(40000)

        assert_almost_equal(1, samples.var(), 1)
        assert_almost_equal(0, samples.mean(), 2)

    def test_normal_n_rvs(self):
        """
        Distributions: Tests for expected number of RVS.
        """
        dist = NormalDistribution(0, 1)
        assert(dist.n_rvs == 1)

    def test_multivar_normal_moments(self):
        """
        Distributions: Checks that the multivariate 
        normal distribtion has the right moments.
        """
        MU = np.array([0,1])
        COV = np.array([[1,0.2],[0.2,2]])
        dist = MultivariateNormalDistribution(MU, COV)

        samples = dist.sample(100000)

        assert_almost_equal(COV, np.cov(samples[:,0],samples[:,1]), 1)
        assert_almost_equal(MU, np.mean(samples, axis=0), 2)
    
    def test_multivar_normal_n_rvs(self):
        """
        Distributions: Tests for expected number of RVS.
        """
        MU = np.array([0,1])
        COV = np.array([[1,0.2],[0.2,2]])
        dist = MultivariateNormalDistribution(MU, COV)
        assert(dist.n_rvs == 2)

class TestUniformDistribution(DerandomizedTestCase):
    """
    Tests ``UniformDistribution``
    """

    ## TEST METHODS ##

    def test_univ_uniform_range(self):
        """
        Distributions: Checks that the univariate uniform dist obeys limits.
        """
        for lower, upper in [(0, 1), (-1, 1), (-1, 5)]:
            dist = UniformDistribution([lower, upper])

            samples = dist.sample(1000)
            assert np.all(samples >= lower)
            assert np.all(samples <= upper)

    def test_univ_uniform_moments(self):
        """
        Distributions: Checks that the univ. uniform dist. has the right moments.
        """
        dist = UniformDistribution([[0, 1]])
        samples = dist.sample(10000)

        # We use low-precision checks here, since the error goes as 1/sqrt{N}.
        # Determinism helps us be sure that once we pass, we'll keep passing,
        # but it does nothing to make the moments accurate.
        assert_almost_equal(1 / 12, samples.var(), 2)
        assert_almost_equal(1 / 2, samples.mean(), 2)

    def test_univ_uniform_n_rvs(self):
        """
        Distributions: Tests for expected number of RVS.
        """
        dist = UniformDistribution([[0, 1]])
        assert(dist.n_rvs == 1)

    def test_uniform_shape(self):
        """
        Distributions: Checks that the multivar. uni. dist has the right shape.
        """
        dist = UniformDistribution([[0, 1], [0, 2], [0, 3]])
        assert dist.sample(100).shape == (100, 3)

    def test_uniform_n_rvs(self):
        """
        Distributions: Tests for expected number of RVS.
        """
        dist = UniformDistribution([[0, 1], [0, 2], [0, 3]])
        assert(dist.n_rvs == 3)

class TestConstantDistribution(DerandomizedTestCase):
    """
    Tests ``ConstantDistribution``
    """

    ## TEST METHODS ##

    def test_constant(self):
        """
        Distributions: Checks that the constant distribution is constant.
        """
        dist = ConstantDistribution([1, 2, 3])
        samples = dist.sample(100)

        assert samples.shape == (100, 3)
        assert np.all(samples[:, 0] == 1)
        assert np.all(samples[:, 1] == 2)
        assert np.all(samples[:, 2] == 3)

    def test_constant_n_rvs(self):
        """
        Distributions: Tests for expected number of RVS.
        """
        dist = ConstantDistribution([1, 2, 3])
        assert(dist.n_rvs == 3)

class TestBetaDistributions(DerandomizedTestCase):
    """
    Tests ``BetaDistribution`` and ``BetaBinomialDistribution``
    """

    ## TEST METHODS ##

    def test_beta_moments(self):
        """
        Distributions: Checks that the beta distribution has the right
        moments, with either of the two input formats
        """
        alpha, beta = 10, 42
        mean = alpha / (alpha + beta)
        var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))

        dist = BetaDistribution(alpha=alpha,beta=beta)
        samples = dist.sample(100000)

        assert samples.shape == (100000,1)
        assert_almost_equal(samples.mean(), mean, 2)
        assert_almost_equal(samples.var(), var, 2)

        dist = BetaDistribution(mean=mean,var=var)
        samples = dist.sample(100000)

        assert samples.shape == (100000,1)
        assert_almost_equal(samples.mean(), mean, 2)
        assert_almost_equal(samples.var(), var, 2)

    def test_beta_n_rvs(self):
        """
        Distributions: Tests for expected number of RVS.
        """
        dist = BetaDistribution(alpha=10,beta=42)
        assert(dist.n_rvs == 1)

    def test_betabinomial_moments(self):
        """
        Distributions: Checks that the beta-binomial distribution has the right
        moments, with either of the two input formats
        """
        n = 10
        alpha, beta = 10, 42
        mean = n * alpha / (alpha + beta)
        var = n * alpha * beta * (alpha + beta + n) / ((alpha + beta) ** 2 * (alpha + beta + 1))

        dist = BetaBinomialDistribution(n, alpha=alpha,beta=beta)
        samples = dist.sample(1000000)

        assert samples.shape == (1000000,1)
        assert_almost_equal(samples.mean(), mean, 1)
        assert_almost_equal(samples.var(), var, 1)

        dist = BetaBinomialDistribution(n, mean=mean,var=var)
        samples = dist.sample(1000000)

        assert samples.shape == (1000000,1)
        assert_almost_equal(samples.mean(), mean, 1)
        assert_almost_equal(samples.var(), var, 1)

    def test_betabinomial_n_rvs(self):
        """
        Distributions: Tests for expected number of RVS.
        """
        dist = BetaBinomialDistribution(10, alpha=10,beta=42)
        assert(dist.n_rvs == 1)

class TestGammaDistribution(DerandomizedTestCase):
    """
    Tests ``GammaDistribution``
    """

    ## TEST METHODS ##

    def test_gamma_moments(self):
        """
        Distributions: Checks that the gamma distribution has the right
        moments, with either of the two input formats
        """
        alpha, beta = 10, 42
        mean = alpha / beta
        var = alpha / beta ** 2

        dist = GammaDistribution(alpha=alpha,beta=beta)
        samples = dist.sample(100000)

        assert samples.shape == (100000,1)
        assert_almost_equal(samples.mean(), mean, 2)
        assert_almost_equal(samples.var(), var, 2)

        dist = GammaDistribution(mean=mean,var=var)
        samples = dist.sample(100000)

        assert samples.shape == (100000,1)
        assert_almost_equal(samples.mean(), mean, 2)
        assert_almost_equal(samples.var(), var, 2)
    
    def test_gamma_n_rvs(self):
        """
        Distributions: Tests for expected number of RVS.
        """
        dist = GammaDistribution(alpha=10,beta=42)
        assert(dist.n_rvs == 1)

class TestProductDistribution(DerandomizedTestCase):
    """
    Tests ``ProductDistribution``
    """

    ## TEST METHODS ##

    def test_product_moments(self):
        """
        Distributions: Checks that product distributions
        have the right moments.
        """

        dist1 = NormalDistribution(0,1)
        dist2 = MultivariateNormalDistribution(np.array([1,2]),np.array([[2,0],[0,3]]))
        dist = ProductDistribution(dist1, dist2)

        samples = dist.sample(100000)

        assert_almost_equal(np.round(np.mean(samples, axis=0)), np.array([0,1,2]))
        assert_almost_equal(np.round(np.var(samples[:,0])), 1)
        assert_almost_equal(np.round(np.var(samples[:,1])), 2)
        assert_almost_equal(np.round(np.var(samples[:,2])), 3)

    def test_product_n_rvs(self):
        """
        Distributions: Tests for expected number of RVS.
        """
        dist1 = NormalDistribution(0,1)
        dist2 = MultivariateNormalDistribution(np.array([1,2]),np.array([[2,0],[0,3]]))
        dist = ProductDistribution(dist1, dist2)
        assert(dist.n_rvs == 3)

class TestMixtureDistribution(DerandomizedTestCase):
    """
    Tests ``MixtureDistribution``
    """

    ## TEST METHODS ##

    def test_mixture_moments(self):
        """
        Distributions: Checks that MixtureDistributions
        has the correct mean value for the normal 
        distrubution under both input formats.
        """
        weights = np.array([0.25, 0.25, 0.5])
        means = np.array([1,2,3])
        vars = np.array([.5, .2, .8])

        dist_list = [
            NormalDistribution(means[idx], vars[idx])
            for idx in range(3)
        ]

        # Test both input formats
        mix1 = MixtureDistribution(weights, dist_list)
        mix2 = MixtureDistribution(weights, NormalDistribution, 
            dist_args=np.vstack([means,vars]).T)
        # Also test with kwargs
        mix3 = MixtureDistribution(weights, NormalDistribution, 
            dist_args=np.vstack([means,vars]).T,
            dist_kw_args={'trunc': np.vstack([means-vars/5,means+vars/5]).T})
        # Also test without the shuffle
        mix4 = MixtureDistribution(weights, dist_list, shuffle=False)
        
        s1 = mix1.sample(150000)
        s2 = mix2.sample(150000)
        s3 = mix3.sample(150000)
        s4 = mix4.sample(150000)

        # The mean should be the weighted means.
        assert_almost_equal(s1.mean(), np.dot(weights, means), 2)
        assert_almost_equal(s2.mean(), np.dot(weights, means), 2)
        assert_almost_equal(s3.mean(), np.dot(weights, means), 2)
        assert_almost_equal(s4.mean(), np.dot(weights, means), 2)

        # The variance should be given by the law of total variance
        assert_almost_equal(
            np.var(s1),
            np.dot(weights, vars) + np.dot(weights, means**2) - np.dot(weights, means)**2,
            1
        )
        assert_almost_equal(
            np.var(s2),
            np.dot(weights, vars) + np.dot(weights, means**2) - np.dot(weights, means)**2,
            1
        )
        # Skip the variance test for s3 because truncation messes with it.
        assert_almost_equal(
            np.var(s4),
            np.dot(weights, vars) + np.dot(weights, means**2) - np.dot(weights, means)**2,
            1
        )

    def test_mixture_n_rvs(self):
        """
        Distributions: Tests for expected number of RVS.
        """
        weights = np.array([0.25, 0.25, 0.5])
        means = np.array([1,2,3])
        vars = np.array([.5, .2, .8])
        dist_list = [
            NormalDistribution(means[idx], vars[idx])
            for idx in range(3)
        ]
        dist = MixtureDistribution(weights, dist_list)
        assert(dist.n_rvs == 1)

        weights = np.array([0.25, 0.25, 0.5])
        means = np.array([[1,0],[2,0],[3,1]])
        vars = np.array([[[1,0.2],[0.2,2]],[[3,0.2],[0.2,2]],[[2,0.2],[0.2,2]]])
        dist_list = [
            MultivariateNormalDistribution(means[idx], vars[idx])
            for idx in range(3)
        ]
        dist = MixtureDistribution(weights, dist_list)
        assert(dist.n_rvs == 2)


