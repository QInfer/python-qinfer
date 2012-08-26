#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# SMC.py: Sequential Monte Carlo module
##
# © 2012 Chris Ferrie (csferrie@gmail.com) and
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

## FEATURES ####################################################################

from __future__ import division

## ALL #########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
    'SMCUpdater'
]

## IMPORTS #####################################################################

import numpy as np
import scipy.linalg as la

## CLASSES #####################################################################

class SMCUpdater(object):
    r"""
    Creates a new Sequential Monte carlo updater

    :param Model model: Model whose parameters are to be inferred.
    :param int n_particles: The number of particles to be used in the particle approximation.
    :param Distribution prior: A representation of the prior distribution.
    :param float resample_a: Specifies the parameter :math:`a` to be used in when resampling.
    :param float resample_thresh: Specifies the threshold for :math:`N_{\text{ess}}` to decide when to resample.
    """
    def __init__(self,
            model, n_particles, prior,
            resample_a=0.98, resample_thresh=0.5
            ):

        self._resample_count = 0

        self.model = model
        self.n_particles = n_particles
        self.prior = prior
        self.resample_a = resample_a
        self.resample_h = np.sqrt(1 - resample_a**2)
        self.resample_thresh = resample_thresh        
        
        self.particle_locations = np.zeros((n_particles, model.n_modelparams))
        self.particle_weights = np.ones((n_particles,)) / n_particles
        
        for idx_particle in xrange(n_particles):
            self.particle_locations[idx_particle, :] = prior.sample()

    ## PROPERTIES ##############################################################
            
    @property
    def resample_count(self):
        # TODO: docstring
        # We wrap this in a property to prevent external resetting and to enable
        # a docstring.
        return self._resample_count
            
    @property
    def n_ess(self):
        """
        Estimates the effective sample size (ESS) of the current distribution
        over model parameters.
        
        :return float: The effective sample size, given by :math:`1/\sum_i w_i^2`.
        """
        return 1 / (sum(self.particle_weights**2))

    def hypothetical_update(self, outcome, expparams):
        """
        Produces the particle weights for the posterior of a hypothetical
        experiment.
        
        :param int outcome: Integer index of the outcome of the hypothetical experiment.
            TODO: Fix this to take an array-like of ints as well.
        :param expparams: TODO
       
        :type weights: ndarray, shape (n_outcomes, n_expparams, n_particles)
        :param weights: Weights assigned to each particle in the posterior
            distribution :math:`\Pr(\omega | d)`.
        """
        
        # It's "hypothetical", don't want to overwrite old weights yet!
        weights = np.copy(self.particle_weights)
        locs = self.particle_locations
        
        # Check if we have a single outcome or an array.
        if not isinstance(outcome, np.ndarray):
            outcome = np.array([outcome])
        
        # update the weights sans normalization
        # Rearrange so that likelihoods have shape (outcomes, experiments, models).
        # This makes the multiplication with weights (shape (models,)) make sense.
        L = self.model.likelihood(outcome, locs, expparams).transpose([0, 2, 1])
        weights = weights * L
        
        # normalize
        return weights / np.sum(weights, axis=2)[..., np.newaxis]
            # Note that newaxis is needed to align the two matrices.
            # This introduces a length-1 axis for the particle number,
            # so that the normalization is broadcast over all particles.
    
    def update(self, outcome, expparams):
        """
        Given an experiment and an outcome of that experiment, updates the
        posterior distribution to reflect knowledge of that experiment.
        
        After updating, resamples the posterior distribution if necessary.
        
        :param int outcome: Index of the outcome of the experiment that was performed.
        :param expparams: TODO
        """
        
        # Since hypothetical_update returns an array indexed by
        # [outcome, experiment, particle], we need to strip off those two
        # indices first.
        self.particle_weights = self.hypothetical_update(outcome, expparams)[0, 0, :]
        
        if self.n_ess < self.n_particles * self.resample_thresh:
            self.resample()
            pass
            
    def resample(self):
        """
        Resample the particles according to algorithm given in 
        Liu and West (2000)
        """
        
        self._resample_count += 1
        
        # parameters in the Liu and West algorithm
        mean, cov = self.est_mean(), self.est_covariance_mtx()
        a, h = self.resample_a, self.resample_h
        S = np.real(h * la.sqrtm(cov))
        n_mp = self.model.n_modelparams
        
        new_locs = np.empty(self.particle_locations.shape)        
        cumsum_weights = np.cumsum(self.particle_weights)
        
        for idx_particle in xrange(self.particle_locations.shape[0]):
            while True:
                # Draw j with probability self.particle_weights[j].
                j = np.argmax(np.random.random() <= cumsum_weights)
                
                # Set mu_i to a x_j + (1 - a) mu.
                mu_i = a * self.particle_locations[j, :] + (1 - a) * mean
                
                # Draw x_i from N(mu_i, S).
                new_locs[idx_particle, :] = mu_i + np.dot(S, np.random.randn(n_mp))
                
                # Set w_i to uniform (done below).
                
                # Check that we're valid, and if so, break out of the inner loop.
                if self.model.is_model_valid(new_locs[idx_particle, :]):
                    break


        # Now we reset the weights to be uniform, letting the density of
        # particles represent the information that used to be stored in the
        # weights.
        self.particle_weights[:] = (1/self.n_particles)
        self.particle_locations = new_locs
        
    ## ESTIMATION METHODS ######################################################
    
    def est_mean(self):
        return np.sum(
            # We need the particle index to be the rightmost index, so that
            # the two arrays align on the particle index as opposed to the
            # modelparam index.
            self.particle_weights * self.particle_locations.transpose([1, 0]),
        axis=1)
        
    def est_covariance_mtx(self):
        mu = self.est_mean()
        xs = self.particle_locations.transpose([1, 0])
        ws = self.particle_weights
        
        return (
            np.sum(
                ws * xs[:, np.newaxis, :] * xs[np.newaxis, :, :],
                axis=2
                )
            ) - np.dot(mu[..., np.newaxis], mu[np.newaxis, ...])
            
        
    
