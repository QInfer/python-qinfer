#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_abstract_model.py: Checks that Model works properly.
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
from numpy.testing import assert_equal, assert_almost_equal, assert_array_less

from qinfer.tests.base_test import DerandomizedTestCase
from qinfer.abstract_model import (
    Model)
from qinfer import ScoreMixin, SimplePrecessionModel, UniformDistribution

from qinfer.smc import SMCUpdater,SMCUpdaterBCRB

# replace analytical score with numerical
class NumericalSimplePrecessionModel(ScoreMixin, SimplePrecessionModel):
    pass



class TestSMCUpdater(DerandomizedTestCase):
	# True model parameter for test
	MODELPARAMS = np.array([1,])
	TEST_EXPPARAMS = np.linspace(1.,10.,100,dtype=np.float)
	PRIOR = UniformDistribution([[0,2]])
	N_PARTICLES = 10000

	TEST_TARGET_COV = np.array([[0.01]])

	def setUp(self):

		super(TestSMCUpdater,self).setUp()
		self.precession_model = SimplePrecessionModel()
		self.num_precession_model = NumericalSimplePrecessionModel() 
		self.expparams = TestSMCUpdater.TEST_EXPPARAMS.reshape(-1,1)
		self.outcomes = self.precession_model.simulate_experiment(TestSMCUpdater.MODELPARAMS,
				TestSMCUpdater.TEST_EXPPARAMS,repeat=1 ).reshape(-1,1)

		self.updater = SMCUpdater(self.precession_model,
				TestSMCUpdater.N_PARTICLES,TestSMCUpdater.PRIOR)
		self.updater_bayes = SMCUpdaterBCRB(self.precession_model,
				TestSMCUpdater.N_PARTICLES,TestSMCUpdater.PRIOR,adaptive=True)
		self.num_updater = SMCUpdater(self.num_precession_model,
				TestSMCUpdater.N_PARTICLES,TestSMCUpdater.PRIOR)
		self.num_updater_bayes = SMCUpdaterBCRB(self.num_precession_model,
				TestSMCUpdater.N_PARTICLES,TestSMCUpdater.PRIOR,adaptive=True)


	def test_smc_fitting(self):
		"""
		Checks that the fitters converge on true value on simple precession_model. Is a stochastic
		test but I ran 100 times and there were no fails, with these parameters.
		"""

		self.updater.batch_update(self.outcomes,self.expparams)
		self.updater_bayes.batch_update(self.outcomes,self.expparams)
		self.num_updater.batch_update(self.outcomes,self.expparams)
		self.num_updater_bayes.batch_update(self.outcomes,self.expparams)

		#Assert that models have learned true model parameters from data 
		#test means
		assert_almost_equal(self.updater.est_mean(),TestSMCUpdater.MODELPARAMS,2)
		assert_almost_equal(self.updater_bayes.est_mean(),TestSMCUpdater.MODELPARAMS,2)
		assert_almost_equal(self.num_updater.est_mean(),TestSMCUpdater.MODELPARAMS,2)
		assert_almost_equal(self.num_updater_bayes.est_mean(),TestSMCUpdater.MODELPARAMS,2)


		#Assert that covariances have been reduced below thresholds
		#test covs 
		assert_array_less(self.updater.est_covariance_mtx(),TestSMCUpdater.TEST_TARGET_COV)
		assert_array_less(self.updater_bayes.est_covariance_mtx(),TestSMCUpdater.TEST_TARGET_COV)
		assert_array_less(self.num_updater.est_covariance_mtx(),TestSMCUpdater.TEST_TARGET_COV)
		assert_array_less(self.num_updater_bayes.est_covariance_mtx(),TestSMCUpdater.TEST_TARGET_COV)

	def test_bim(self):
		"""
		Checks that the fitters converge on true value on simple precession_model. Is a stochastic
		test but I ran 100 times and there were no fails, with these parameters.
		"""
		bim_currents = []
		num_bim_currents = []
		bim_adaptives = []
		num_bim_adaptives = []

		#track bims throughout experiments
		for i in range(self.outcomes.shape[0]):			
			self.updater_bayes.update(self.outcomes[i],self.expparams[i])
			self.num_updater_bayes.update(self.outcomes[i],self.expparams[i])

			bim_currents.append(self.updater_bayes.current_bim)
			num_bim_currents.append(self.num_updater_bayes.current_bim)
			bim_adaptives.append(self.updater_bayes.adaptive_bim)
			num_bim_adaptives.append(self.num_updater_bayes.adaptive_bim)

		bim_currents = np.array(bim_currents)
		num_bim_currents = np.array(num_bim_currents)
		bim_adaptives = np.array(bim_adaptives)
		num_bim_adaptives = np.array(num_bim_adaptives)

		#compare numerical and analytical bims 
		assert_almost_equal(bim_currents,num_bim_currents,2)
		assert_almost_equal(bim_adaptives,num_bim_adaptives,2)

		#verify that array copying of properties is working
		assert not np.all(bim_currents == bim_currents[0,...])
		assert not np.all(num_bim_currents == num_bim_currents[0,...])
		assert not np.all(bim_adaptives == bim_adaptives[0,...])
		assert not np.all(num_bim_adaptives == num_bim_adaptives[0,...])


		#verify that BCRB is approximately reached 
		assert_almost_equal(self.updater_bayes.est_covariance_mtx(),np.linalg.inv(self.updater_bayes.current_bim),2)
		assert_almost_equal(self.updater_bayes.est_covariance_mtx(),np.linalg.inv(self.updater_bayes.adaptive_bim),2)
		assert_almost_equal(self.num_updater_bayes.est_covariance_mtx(),np.linalg.inv(self.updater_bayes.current_bim),2)
		assert_almost_equal(self.num_updater_bayes.est_covariance_mtx(),np.linalg.inv(self.updater_bayes.adaptive_bim),2)
