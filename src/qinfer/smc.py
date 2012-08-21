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

## IMPORTS ##

import numpy as np

class SMCUpdater(object):
    """
    Creates a new Sequential Monte carlo updater

    Parameters
    -----------
    model : Model
        Model whose parameters are to be inferred.
    n_particles : int
        The number of particles to be used in the particle approximation.
    prior : Distribution
        A representation of the prior distribution.
    resample_a : float, Optional (default = 0.98)
        Specifies the parameter :math:`a` to be used in when resampling.
    resample_thresh : float, Optional (default = 0.5)
        Specifies the threshold for :math:`n_ess` to decide when to resample.
    """
    def __init__(self,
            model, n_particles, prior,
            resample_a=0.98, resample_thresh=0.5
            ):

        self.model = model
        self.n_particles = n_particles
        self.prior = prior
        self.resample_a = resample_a
        self.resample_thresh = resample_thresh        
        
        self.particle_locs = np.zeros((n_particles,))
        self.particle_weights = np.ones((n_particles,)) / n_particles
        
        for idx_particle in xrange(n_particles):
            self.particle_locations[idx_particle, :] = prior.sample()
            
    def n_ess(self):
        """
        Estimates the effective sample size (ESS) of the current distribution
        over model parameters.
        
        Returns
        -------
        N : float
            The effective sample size, given by :math:`1/\sum_i w_i^2`.
        """
        return 1 / (sum(self.particle_weights**2))
