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
import scipy.stats
from numpy.testing import assert_equal, assert_almost_equal, assert_array_less

from qinfer.tests.base_test import DerandomizedTestCase
from qinfer.utils import assert_sigfigs_equal
from qinfer.distributions import *

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

class TestSlantedNormalDistribution(DerandomizedTestCase):
    """
    Tests ``SlantedNormalDistribution``
    """

    ## TEST METHODS ##

    #TODO
    def test_slantednormal_moments(self):
        """
        Distributions: Checks that the slanted normal
        distribution has the right moments.
        """
        ranges = [[0,1],[0,2],[2,3]]
        weight = 2
        dist = SlantedNormalDistribution(ranges=ranges, weight=weight)

        samples = dist.sample(150000)

        assert_sigfigs_equal(
            np.mean(np.array(ranges), axis=1),
            np.mean(samples, axis=0),
        1)
        assert_sigfigs_equal(1/12+4, samples[:,0].var(), 1)
        assert_sigfigs_equal(4/12+4, samples[:,1].var(), 1)
        assert_sigfigs_equal(1/12+4, samples[:,2].var(), 1)

class TestLogNormalDistribution(DerandomizedTestCase):
    """
    Tests ``LogNormalDistribution``
    """

    ## TEST METHODS ##

    def test_lognormal_moments(self):
        """
        Distributions: Checks that the log normal
        distribution has the right moments.
        """
        mu, sig = 3, 2
        dist = LogNormalDistribution(mu=mu, sigma=sig)

        samples = dist.sample(150000)

        assert_sigfigs_equal(
            scipy.stats.lognorm.mean(1,3,2),
            samples.mean(),
            1)
        assert_sigfigs_equal(
            np.round(scipy.stats.lognorm.var(1,3,2)),
            np.round(samples.var()),
            1)

    def test_lognormal_n_rvs(self):
        """
        Distributions: Tests for expected number of RVS.
        """
        dist = LogNormalDistribution(mu=3, sigma=2)
        assert(dist.n_rvs == 1)


class TestUniformDistribution(DerandomizedTestCase):
    """
    Tests ``UniformDistribution`` and ``DiscreteUniformDistribution``
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

    def test_discrete_uniform_moments(self):
        """
        Distributions: Checks that the discrete uniform dist. has the right moments.
        """
        dist = DiscreteUniformDistribution(5)
        samples = dist.sample(200000).astype(float)

        assert_sigfigs_equal((2**10-1)/12, np.var(samples), 1)
        assert_sigfigs_equal(16, np.mean(samples), 1)

    def test_discrete_uniform_n_rvs(self):
        """
        Distributions: Tests for expected number of RVS.
        """
        dist = DiscreteUniformDistribution(5)
        assert(dist.n_rvs == 1)



class TestMVUniformDistribution(DerandomizedTestCase):
    """
    Tests ``MVUniformDistribution``
    """

    ## TEST METHODS ##

    def test_mvuniform_moments(self):
        """
        Distributions: Checks that ``MVUniformDistribution`` has the right moments.
        """
        dist = MVUniformDistribution(dim=6)
        samples = dist.sample(100000)

        assert_sigfigs_equal(5/(36*7), samples[:,3].var(), 2)
        assert_sigfigs_equal(np.array([1/6]*6), np.mean(samples, axis=0), 2)

    def test_mvuniform_n_rvs(self):
        """
        Distributions: Tests for expected number of RVS.
        """
        dist = MVUniformDistribution(dim=3)
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

class TestSingleSampleMixin(DerandomizedTestCase):
    """
    Tests ``SingleSampleMixin``
    """

    ## TEST METHODS ##

    def test_single_sample_mixin(self):
        """
        Distributions: Tests that subclassing from
        SingleSampleMixin works.
        """

        class TestDist(SingleSampleMixin, Distribution):
            def __init__(self, dist):
                super(TestDist, self).__init__()
                self._dist = dist
            @property
            def n_rvs(self):
                return self._dist.n_rvs
            def _sample(self):
                return self._dist.sample(n=1)

        dist1 = TestDist(NormalDistribution(0,1))
        dist2 = TestDist(MultivariateNormalDistribution(np.array([1,2]),np.array([[2,0],[0,3]])))

        sample1 = dist1.sample(500)
        sample2 = dist2.sample(500)

        assert(sample1.shape == (500,1))
        assert(sample2.shape == (500,2))

        assert_almost_equal(np.round(np.mean(sample1,axis=0)), 0)
        assert_almost_equal(np.round(np.mean(sample2,axis=0)), np.array([1,2]))

class TestHaarUniform(DerandomizedTestCase):
    """
    Tests ``HaarUniform``
    """

    ## TEST METHODS ##

    def test_haar_state_mean(self):
        """
        Distributions: Checks that HaarUniform
        has the correct mean.
        """

        dist = HaarUniform()
        samples = dist.sample(1000)

        x = np.mean(samples[:,0]) * np.array([[0,1],[1,0]])
        y = np.mean(samples[:,1]) * np.array([[0,-1j],[1j,0]])
        z = np.mean(samples[:,2]) * np.array([[1,0],[0,1]])

        rho = x + y + z

        assert_almost_equal(rho, np.zeros((2,2)), 2)

    def test_haar_state_n_rvs(self):
        """
        Distributions: Tests for expected number of RVS.
        """

        dist = HaarUniform()
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

class TestParticleDistribution(DerandomizedTestCase):
    """
    Tests ``ParticleDistribution``
    """

    ## TEST METHODS ##

    def test_init(self):
        """
        Distributions: Checks that ParticleDistributions
        initialized correctly in different circumstances.
        """
        dim = 5
        n_particles = 100
        # note that these weights are not all positive!
        particle_weights = np.random.randn(dim)
        particle_weights = particle_weights
        particle_locations = np.random.rand(n_particles, dim)

        dist1 = ParticleDistribution(n_mps=dim)
        dist2 = ParticleDistribution(
            particle_weights=particle_weights,
            particle_locations=particle_locations
        )

        assert(dist1.n_particles == 1)
        assert(dist1.n_rvs == dim)
        assert_almost_equal(dist1.sample(3), np.zeros((3, dim)))
        assert_almost_equal(np.sum(dist1.particle_weights), 1)

        assert(dist2.n_particles == n_particles)
        assert(dist2.n_rvs == dim)
        assert(dist2.sample(3).shape == (3,dim))
        # the following demands that ParticleDistribution
        # retcifies and normalizes whichever weights it is given
        assert_almost_equal(np.sum(dist2.particle_weights), 1)

    def test_ness(self):
        """
        Distributions: Tests the n_ess property of the
        ParticleDistribution.
        """

        dim = 5
        n_particles = 100
        particle_weights1 = np.ones(n_particles) / n_particles
        particle_weights2 = np.zeros(n_particles)
        particle_weights2[0] = 1
        particle_weights3 = np.random.rand(dim)
        particle_weights3 = particle_weights3 / np.sum(particle_weights3)
        particle_locations = np.random.rand(n_particles, dim)

        dist1 = ParticleDistribution(
            particle_weights=particle_weights1,
            particle_locations=particle_locations
        )
        dist2 = ParticleDistribution(
            particle_weights=particle_weights2,
            particle_locations=particle_locations
        )
        dist3 = ParticleDistribution(
            particle_weights=particle_weights3,
            particle_locations=particle_locations
        )

        assert_almost_equal(dist1.n_ess, n_particles)
        assert_almost_equal(dist2.n_ess, 1)
        assert(dist3.n_ess < n_particles and dist3.n_ess > 1)

    def test_moments(self):
        """
        Distributions: Tests the moment function (est_mean, etc)
        of ParticleDistribution.
        """

        dim = 5
        n_particles = 100000
        # draw particles from a randomly chosen mutivariate normal
        mu = np.random.randn(dim)
        cov = np.random.randn(dim,dim)
        cov = np.dot(cov,cov.T)
        particle_locations = np.random.multivariate_normal(mu, cov, n_particles)
        particle_weights = np.random.rand(n_particles)

        dist = ParticleDistribution(
            particle_weights=particle_weights,
            particle_locations=particle_locations
        )

        assert_sigfigs_equal(mu, dist.est_mean(), 1)
        assert_almost_equal(dist.est_meanfn(lambda x: x**2),np.diag(cov) + mu**2, 0)
        assert(np.linalg.norm(dist.est_covariance_mtx() - cov) < 0.5)

    def test_entropy(self):
        """
        Distributions: Tests the entropy and related functions of
        ParticleDistributions.
        """

        dim = 3
        n_particles = 100
        # draw particles from a randomly chosen mutivariate normal
        mu = np.random.randn(dim)
        cov = np.random.randn(dim,dim)
        cov = np.dot(cov,cov.T)
        particle_locations = np.random.multivariate_normal(mu, cov, n_particles)
        particle_weights1 = np.ones(n_particles)
        particle_weights2 = np.random.rand(n_particles)

        dist1 = ParticleDistribution(
            particle_weights=particle_weights1,
            particle_locations=particle_locations
        )
        dist2 = ParticleDistribution(
            particle_weights=particle_weights2,
            particle_locations=particle_locations
        )

        assert_almost_equal(dist1.est_entropy(), np.log(n_particles))
        #TODO: test that est_kl_divergence does more than not fail
        dist1.est_kl_divergence(dist2)

    def test_clustering(self):
        """
        Distributions: Tests that clustering works.
        """

        dim = 3
        n_particles = 1000
        # make two multivariate normal clusters
        mu1 = 50+np.zeros(dim)
        mu2 = -50+np.zeros(dim)
        cov = np.random.randn(dim,dim)
        cov = np.dot(cov,cov.T)
        particle_locations = np.concatenate([
            np.random.multivariate_normal(mu1, cov, int(n_particles/2)),
            np.random.multivariate_normal(mu2, cov, int(n_particles/2))
        ])
        particle_weights = np.ones(n_particles)

        dist = ParticleDistribution(
            particle_weights=particle_weights,
            particle_locations=particle_locations
        )

        # TODO: do more than check these don't fail (I didn't have time
        # to figure out the undocumented code.)
        dist.est_cluster_moments()
        dist.est_cluster_covs()
        dist.est_cluster_metric()



class TestInterpolatedUnivariateDistribution(DerandomizedTestCase):
    """
    Tests ``InterpolatedUnivariateDistribution``
    """

    def test_interp_moments(self):
        """
        Distributions: Checks that the interpolated distribution
        has the right moments.
        """

        # Interpolate the normal distribution because we
        # know the moments
        dist = InterpolatedUnivariateDistribution(
            scipy.stats.norm.pdf, 1, 1500
        )

        samples = dist.sample(40000)

        assert_almost_equal(1, samples.var(), 1)
        assert_almost_equal(0, samples.mean(), 1)

    def test_interp_n_rvs(self):
        """
        Distributions: Tests for expected number of RVS.
        """
        dist = InterpolatedUnivariateDistribution(
            scipy.stats.norm.pdf, 1, 1500
        )
        assert(dist.n_rvs == 1)

class TestPostselectedDistribution(DerandomizedTestCase):
    """
    Tests ``PostselectedDistribution``
    """

    def test_postselected_validity(self):
        """
        Distributions: Checks that the postselected
        samples are valid.
        """

        ud = NormalDistribution(0, 1)
        class FakeModel(object):
            def are_models_valid(self, mps):
                return mps >= 0
        dist = PostselectedDistribution(
            ud, FakeModel()
        )

        samples = dist.sample(40000)
        assert_array_less(0, samples)

    def test_postselected_fails(self):
        """
        Distributions: Checks that the postselected
        fails to generate enough points with a
        difficult constraint.
        """

        ud = NormalDistribution(0, 1)
        class FakeModel(object):
            def are_models_valid(self, mps):
                return mps >= 1000
        dist = PostselectedDistribution(
            ud, FakeModel(), 30
        )
        self.assertRaises(
            RuntimeError,
            dist.sample,
            10000
        )

    def test_postselected_n_rvs(self):
        """
        Distributions: Tests for expected number of RVS.
        """
        ud = NormalDistribution(0, 1)
        class FakeModel(object):
            def are_models_valid(self, mps):
                return mps >= 1000
        dist = PostselectedDistribution(
            ud, FakeModel(), 30
        )
        assert(dist.n_rvs == 1)

class TestConstrainedSumDistribution(DerandomizedTestCase):
    """
    Tests ``ConstrainedSumDistribution``
    """

    def test_constrained_sum_constraint(self):
        """
        Distributions: Tests that the contstraint is met in
        the constrained sum distribution.
        """

        unif = UniformDistribution([[0,1],[0,2]])
        dist = ConstrainedSumDistribution(unif, 3)

        samples = dist.sample(1000)

        assert_almost_equal(
            np.sum(samples, axis=1),
            3 * np.ones(1000)
        )

    def test_constrained_sum_moments(self):
        """
        Distributions: Tests that the contstraint is met in
        the constrained sum distribution.
        """

        unif = UniformDistribution([[0,1],[0,1]])
        dist = ConstrainedSumDistribution(unif, 1)

        samples = dist.sample(100000)

        assert_sigfigs_equal(np.array([1/2]*2), np.mean(samples, axis=0), 2)


    def test_constrained_sum_n_rvs(self):
        """
        Distributions: Tests for expected number of RVS.
        """
        unif = UniformDistribution([[0,1],[0,2]])
        dist = ConstrainedSumDistribution(unif, 3)

        assert(dist.n_rvs == 2)
