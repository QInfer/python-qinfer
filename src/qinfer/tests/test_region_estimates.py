#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_region_estimates.py: Checks that computed credible regions are working.
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
from numpy.testing import assert_equal, assert_almost_equal, assert_array_less

from qinfer.abstract_model import FiniteOutcomeModel
from qinfer.tests.base_test import DerandomizedTestCase, MockModel
from qinfer.distributions import MultivariateNormalDistribution, ParticleDistribution
from qinfer.smc import SMCUpdater

## FUNCTIONS ##################################################################

def unique_rows(a):
    """
    Discards duplicate rows.
    """
    # from http://stackoverflow.com/a/16971324/1082565
    ind = np.lexsort(a.T)
    return a[ind[np.concatenate(([True],np.any(a[ind[1:]]!=a[ind[:-1]],axis=1)))]]

## CLASSES ####################################################################

class TestSMCCredibleRegions(DerandomizedTestCase):

    N_PARTICLES = 10000
    N_MPS = 4
    MEAN = np.array([2,3,5,7])
    COV = np.array([[1,0,0,0.5],[0,1,0.2,0],[0,0.2,2,0],[0.5,0,0,1]])
    SLICE = np.s_[:2]

    def test_est_credible_region(self):
        """
        Tests that est_credible_region doesn't fail miserably
        """
        dist = MultivariateNormalDistribution(self.MEAN, self.COV)
        # the model is irrelevant; we just want the updater to have some particles
        # with the desired normal distribution.
        u = SMCUpdater(MockModel(self.N_MPS), self.N_PARTICLES, dist)

        # first check that 0.95 confidence points consume 0.9 confidence points
        points1 = u.est_credible_region(level=0.95)
        points2 = u.est_credible_region(level=0.9)
        assert_almost_equal(
            np.sort(unique_rows(np.concatenate([points1, points2])), axis=0),
            np.sort(points1, axis=0)
        )

        # do the same thing with different slice
        points1 = u.est_credible_region(level=0.95, modelparam_slice=self.SLICE)
        points2 = u.est_credible_region(level=0.9, modelparam_slice=self.SLICE)
        assert_almost_equal(
            np.sort(unique_rows(np.concatenate([points1, points2])), axis=0),
            np.sort(points1, axis=0)
        )

    def test_region_est_hull(self):
        """
        Tests that test_region_est_hull works
        """
        dist = MultivariateNormalDistribution(self.MEAN, self.COV)
        # the model is irrelevant; we just want the updater to have some particles
        # with the desired normal distribution.
        u = SMCUpdater(MockModel(self.N_MPS), self.N_PARTICLES, dist)

        faces, vertices = u.region_est_hull(level=0.95)

        # In this multinormal case, the convex hull surface
        # should be centered at MEAN
        assert_almost_equal(
            np.round(np.mean(vertices, axis=0)),
            np.round(self.MEAN)
        )

        # And a lower level should result in a smaller hull
        # and therefore smaller sample variance
        faces2, vertices2 = u.region_est_hull(level=0.2)
        assert_array_less(np.var(vertices2, axis=0), np.var(vertices, axis=0))

    def test_region_est_ellipsoid(self):
        """
        Tests that region_est_ellipsoid works.
        """

        dist = MultivariateNormalDistribution(self.MEAN, self.COV)
        # the model is irrelevant; we just want the updater to have some particles
        # with the desired normal distribution.
        u = SMCUpdater(MockModel(4), self.N_PARTICLES, dist)

        # ask for a confidence level of 0.5
        A, c = u.region_est_ellipsoid(level=0.5)

        # center of ellipse should be the mean of the multinormal
        assert_almost_equal(np.round(c), self.MEAN, 1)

        # finally, the principal lengths of the ellipsoid
        # should be the same as COV
        _, QA, _ = np.linalg.svd(A)
        _, QC, _ = np.linalg.svd(self.COV)
        QA, QC = np.sqrt(QA), np.sqrt(QC)
        assert_almost_equal(
            QA / np.linalg.norm(QA),
            QC / np.linalg.norm(QC),
            1
        )

    def test_in_credible_region(self):
        """
        Tests that in_credible_region works.
        """

        dist = MultivariateNormalDistribution(self.MEAN, self.COV)
        # the model is irrelevant; we just want the updater to have some particles
        # with the desired normal distribution.
        u = SMCUpdater(MockModel(4), self.N_PARTICLES, dist)

        # some points to test with
        test_points = np.random.multivariate_normal(self.MEAN, self.COV, self.N_PARTICLES)

        # method='pce'
        results = [
            u.in_credible_region(test_points, level=0.9, method='pce'),
            u.in_credible_region(test_points, level=0.84, method='pce'),
            u.in_credible_region(test_points, level=0.5, method='pce'),
        ]
        assert_almost_equal(
            np.array([np.mean(x.astype('float')) for x in results]),
            np.array([0.9, 0.84, 0.5]),
            3
        )

        # method='hpd-hull'
        results1 = [
            u.in_credible_region(test_points, level=0.9, method='hpd-hull'),
            u.in_credible_region(test_points, level=0.84, method='hpd-hull'),
            u.in_credible_region(test_points, level=0.5, method='hpd-hull'),
        ]
        assert_array_less(
            np.array([0.9, 0.84, 0.5]),
            np.array([np.mean(x.astype('float')) for x in results1])
        )

        # method='hpd-mvee'
        results2 = [
            u.in_credible_region(test_points, level=0.9, method='hpd-mvee'),
            u.in_credible_region(test_points, level=0.84, method='hpd-mvee'),
            u.in_credible_region(test_points, level=0.5, method='hpd-mvee'),
        ]
        assert_array_less(
            np.array([0.9, 0.84, 0.5]),
            np.array([np.mean(x.astype('float')) for x in results2])
        )

        # the mvee should be bigger than the convex hull.
        # this passes iff all points in the ellipses are
        # also in the hulls.
        assert_array_less(
            np.hstack([x.astype('float') for x in results1]),
            np.hstack([x.astype('float') for x in results2]) + 0.5
        )

        # check for no failures with slices.
        u.in_credible_region(test_points[:100,self.SLICE], level=0.9, method='pce', modelparam_slice=self.SLICE)
        u.in_credible_region(test_points[:100,self.SLICE], level=0.9, method='hpd-hull', modelparam_slice=self.SLICE)
        u.in_credible_region(test_points[:100,self.SLICE], level=0.9, method='hpd-mvee', modelparam_slice=self.SLICE)

        # check for no failures with single inputs
        assert(u.in_credible_region(test_points[0,:], level=0.9, method='pce').size == 1)
        assert(u.in_credible_region(test_points[0,:], level=0.9, method='hpd-hull').size == 1)
        assert(u.in_credible_region(test_points[0,:], level=0.9, method='hpd-mvee').size == 1)

class TestCredibleRegions(DerandomizedTestCase):

    N_PARTICLES = 10000
    N_MPS = 4
    MEAN = np.array([2,3,5,7])
    COV = np.array([[1,0,0,0.5],[0,1,0.2,0],[0,0.2,2,0],[0.5,0,0,1]])
    SLICE = np.s_[:2]

    def test_est_credible_region(self):
        """
        Tests that est_credible_region doesn't fail miserably
        """
        dist = MultivariateNormalDistribution(self.MEAN, self.COV)
        u = ParticleDistribution(
            particle_locations = dist.sample(self.N_PARTICLES),
            particle_weights = np.ones(self.N_PARTICLES)/self.N_PARTICLES
        )

        # first check that 0.95 confidence points consume 0.9 confidence points
        points1 = u.est_credible_region(level=0.95)
        points2 = u.est_credible_region(level=0.9)
        assert_almost_equal(
            np.sort(unique_rows(np.concatenate([points1, points2])), axis=0),
            np.sort(points1, axis=0)
        )

        # do the same thing with different slice
        points1 = u.est_credible_region(level=0.95, modelparam_slice=self.SLICE)
        points2 = u.est_credible_region(level=0.9, modelparam_slice=self.SLICE)
        assert_almost_equal(
            np.sort(unique_rows(np.concatenate([points1, points2])), axis=0),
            np.sort(points1, axis=0)
        )

    def test_region_est_hull(self):
        """
        Tests that test_region_est_hull works
        """
        dist = MultivariateNormalDistribution(self.MEAN, self.COV)
        u = ParticleDistribution(
            particle_locations = dist.sample(self.N_PARTICLES),
            particle_weights = np.ones(self.N_PARTICLES)/self.N_PARTICLES
        )

        faces, vertices = u.region_est_hull(level=0.95)

        # In this multinormal case, the convex hull surface
        # should be centered at MEAN
        assert_almost_equal(
            np.round(np.mean(vertices, axis=0)),
            np.round(self.MEAN)
        )

        # And a lower level should result in a smaller hull
        # and therefore smaller sample variance
        faces2, vertices2 = u.region_est_hull(level=0.2)
        assert_array_less(np.var(vertices2, axis=0), np.var(vertices, axis=0))

    def test_region_est_ellipsoid(self):
        """
        Tests that region_est_ellipsoid works.
        """

        dist = MultivariateNormalDistribution(self.MEAN, self.COV)
        u = ParticleDistribution(
            particle_locations = dist.sample(self.N_PARTICLES),
            particle_weights = np.ones(self.N_PARTICLES)/self.N_PARTICLES
        )

        # ask for a confidence level of 0.5
        A, c = u.region_est_ellipsoid(level=0.5)

        # center of ellipse should be the mean of the multinormal
        assert_almost_equal(np.round(c), self.MEAN, 1)

        # finally, the principal lengths of the ellipsoid
        # should be the same as COV
        _, QA, _ = np.linalg.svd(A)
        _, QC, _ = np.linalg.svd(self.COV)
        QA, QC = np.sqrt(QA), np.sqrt(QC)
        assert_almost_equal(
            QA / np.linalg.norm(QA),
            QC / np.linalg.norm(QC),
            1
        )

    def test_in_credible_region(self):
        """
        Tests that in_credible_region works.
        """

        dist = MultivariateNormalDistribution(self.MEAN, self.COV)
        u = ParticleDistribution(
            particle_locations = dist.sample(self.N_PARTICLES),
            particle_weights = np.ones(self.N_PARTICLES)/self.N_PARTICLES
        )

        # some points to test with
        test_points = np.random.multivariate_normal(self.MEAN, self.COV, self.N_PARTICLES)

        # method='pce'
        results = [
            u.in_credible_region(test_points, level=0.9, method='pce'),
            u.in_credible_region(test_points, level=0.84, method='pce'),
            u.in_credible_region(test_points, level=0.5, method='pce'),
        ]
        assert_almost_equal(
            np.array([np.mean(x.astype('float')) for x in results]),
            np.array([0.9, 0.84, 0.5]),
            3
        )

        # method='hpd-hull'
        results1 = [
            u.in_credible_region(test_points, level=0.9, method='hpd-hull'),
            u.in_credible_region(test_points, level=0.84, method='hpd-hull'),
            u.in_credible_region(test_points, level=0.5, method='hpd-hull'),
        ]
        assert_array_less(
            np.array([0.9, 0.84, 0.5]),
            np.array([np.mean(x.astype('float')) for x in results1])
        )

        # method='hpd-mvee'
        results2 = [
            u.in_credible_region(test_points, level=0.9, method='hpd-mvee'),
            u.in_credible_region(test_points, level=0.84, method='hpd-mvee'),
            u.in_credible_region(test_points, level=0.5, method='hpd-mvee'),
        ]
        assert_array_less(
            np.array([0.9, 0.84, 0.5]),
            np.array([np.mean(x.astype('float')) for x in results2])
        )

        # the mvee should be bigger than the convex hull.
        # this passes iff all points in the ellipses are
        # also in the hulls.
        assert_array_less(
            np.hstack([x.astype('float') for x in results1]),
            np.hstack([x.astype('float') for x in results2]) + 0.5
        )

        # check for no failures with slices.
        u.in_credible_region(test_points[:100,self.SLICE], level=0.9, method='pce', modelparam_slice=self.SLICE)
        u.in_credible_region(test_points[:100,self.SLICE], level=0.9, method='hpd-hull', modelparam_slice=self.SLICE)
        u.in_credible_region(test_points[:100,self.SLICE], level=0.9, method='hpd-mvee', modelparam_slice=self.SLICE)

        # check for no failures with single inputs
        assert(u.in_credible_region(test_points[0,:], level=0.9, method='pce').size == 1)
        assert(u.in_credible_region(test_points[0,:], level=0.9, method='hpd-hull').size == 1)
        assert(u.in_credible_region(test_points[0,:], level=0.9, method='hpd-mvee').size == 1)
