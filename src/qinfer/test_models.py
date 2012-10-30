#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_models.py: Simple models for testing inference engines.
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

## FEATURES ##

from __future__ import division # Ensures that a/b is always a float.

## IMPORTS ##

import numpy as np

from utils import binomial_pdf

from abstract_model import Model,DifferentiableModel
    
## CLASSES ##

class SimplePrecessionModel(DifferentiableModel):
    
    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return 1
        
    @property
    def expparams_dtype(self):
        return 'float'
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        return True
    
    ## METHODS ##
    
    @staticmethod
    def are_models_valid(modelparams):
        return modelparams > 0
    
    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        return 2
    
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(SimplePrecessionModel, self).likelihood(outcomes, modelparams, expparams)
        
        # Allocating first serves to make sure that a shape mismatch later
        # will cause an error.
        pr0 = np.zeros((modelparams.shape[0], expparams.shape[0]))
        
        arg = np.dot(modelparams, expparams[..., np.newaxis].T) / 2        
        pr0 = np.cos(arg) ** 2
        
        # Now we concatenate over outcomes.
        return Model.pr0_to_likelihood_array(outcomes, pr0)

    def grad_log_likelihood(self, outcome, modelparams, expparams):
        #TODO: vectorize this

        arg = modelparams * expparams / 2        
        return (
            ( expparams / np.tan(arg)) ** (outcome) *
            (-expparams * np.tan(arg)) ** (1-outcome)
        )
## TESTING CODE ################################################################

if __name__ == "__main__":
    from distributions import UniformDistribution
    import smc
    import matplotlib.pyplot as plt

    N_PARTICLES = 1000
    
    prior = UniformDistribution([0,1])
    model = SimplePrecessionModel()
    
    updater = smc.SMCUpdaterBCRB(model, N_PARTICLES, prior,resample_a=.99, resample_thresh=0.5)
        
    # Sample true set of modelparams
    truemp = prior.sample()
    
    # Plot true state and prior
#    fig = plt.figure()
#    particles = updater.particle_locations
#    weights = updater.particle_weights      
#    
#    plt.plot(particles[:,0],weights)
    
    # Get all Bayesian up in here
    n_exp = 100
    for idx_exp in xrange(n_exp):
        thisexp = np.array([np.random.random()],dtype=model.expparams_dtype)
   
        outcome = model.simulate_experiment(truemp, thisexp)
       
        updater.update(outcome, thisexp)
        
#        if np.mod(3*idx_exp,n_exp)==0:
#            fig = plt.figure()
#            
#            particles = updater.particle_locations
#            weights = updater.particle_weights      
#            plt.plot(particles[:,0],weights)

    est_mean = updater.est_mean()
    
    print "True param: {}".format(truemp)    
    print "Est. mean: {}".format(updater.est_mean())
#    print "Est. cov: {}".format(updater.est_covariance_mtx())
    print "Error: {}".format(np.sum(np.abs(truemp[0]-updater.est_mean())**2))
    print "Trace Cov: {}".format(np.trace(updater.est_covariance_mtx()))
    print "Resample count: {}".format(updater.resample_count)
    print "BCRB: {}".format(1/updater.current_bim)
        
    
    plt.show()  
