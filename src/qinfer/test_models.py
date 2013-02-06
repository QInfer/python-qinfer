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

from abstract_model import Model, DifferentiableModel
    
## CLASSES #####################################################################

class SimplePrecessionModel(DifferentiableModel):
    r"""
    Describes the free evolution of a single qubit prepared in the
    :math:`\ket{+}` state under a Hamiltonian :math:`H = \omega \sigma_z / 2`,
    as explored in [GFWC12]_. (TODO: add other citations.)
    """
    
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
        
        # Possibly add a second axis to modelparams.
        if len(modelparams.shape) == 1:
            modelparams = modelparams[..., np.newaxis]
        
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
        
class NoisyCoinModel(Model):
    r"""
    Implements the "noisy coin" model of [FB12]_, where the model parameter
    :math:`p` is the probability of the noisy coin. This model has two
    experiment parameters, :math:`\alpha` and :math:`\beta`, which are the
    probabilities of observing a "0" outcome conditoned on the "true" outcome
    being 0 and 1, respectively. That is, for an ideal coin, :math:`\alpha = 1`
    and :math:`\beta = 0`.
    
    Note that :math:`\alpha` and :math:`\beta` are implemented as experiment
    parameters not because we expect to design over those values, but because
    a specification of each is necessary to honestly describe an experiment
    that was performed.
    """
        
    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return 1
        
    @property
    def expparams_dtype(self):
        return [('alpha','float'), ('beta','float')]
    
    @property
    def is_n_outcomes_constant(self):
        return True
    
    ## METHODS ##
    
    @staticmethod
    def are_models_valid(modelparams):
        return np.logical_and(modelparams.all(axis=1) >= 0,modelparams.all(axis=1) <= 1)
    
    def n_outcomes(self, expparams):
        return 2
    
    def likelihood(self, outcomes, modelparams, expparams):
        # Unpack alpha and beta.
        a = expparams['alpha']
        b = expparams['beta']
        
        # Find the probability of getting a "0" outcome.
        pr0 = modelparams * a + (1 - modelparams) * b
        
        # Concatenate over outcomes.
        return Model.pr0_to_likelihood_array(outcomes, pr0)
        
## TESTING CODE ################################################################

# TODO: the following needs cleaned up quite badly.
if __name__ == "__main__":
    from distributions import UniformDistribution
    import smc
    import matplotlib.pyplot as plt
    from scipy.stats.kde import gaussian_kde
    from copy import copy
    
    N_PARTICLES = 10000
    
    prior = UniformDistribution([0,1])
    model = SimplePrecessionModel()

# To test BCRB uncomment all BIM lines   
#    updater = smc.SMCUpdaterBCRB(model, N_PARTICLES, prior,resample_a=.99, resample_thresh=0.5)
        
# To test ABC
    updaterEXACT = smc.SMCUpdater(model, N_PARTICLES, prior)
    updaterABC = smc.SMCUpdaterABC(model, N_PARTICLES, prior, abc_tol = 0, abc_sim = 1e2)

    # Sample true set of modelparams
    truemp = prior.sample()
    
    # Plot true state and prior
    fig = plt.figure()
    
    particles = updaterEXACT.particle_locations
    weights = updaterEXACT.particle_weights      
    particlesABC = updaterABC.particle_locations
    weightsABC = updaterABC.particle_weights
    
    #this is shameful hack to get the Kernal estimate
    temp = copy(updaterEXACT)
    temp.resample
    
    pdf = gaussian_kde(temp.particle_locations[:,0])
    
    x = np.linspace(0,1,1000)
    plt.plot(particles[:,0],weights, '.')
    plt.plot(x,pdf(x),'b')

    temp = copy(updaterABC)
    temp.resample
    
    pdf = gaussian_kde(temp.particle_locations[:,0])
    plt.plot(x,pdf(x),'r')

    plt.plot(particlesABC[:,0],weightsABC, '.r')
    plt.plot(truemp,0,'g')

    # Get all Bayesian up in here
    n_exp = 1000
    
    # theoretical BIM
#    BIM = 0    
    
    for idx_exp in xrange(n_exp):
        thisexp = np.array([np.random.random()],dtype=model.expparams_dtype)
        
#        BIM += thisexp**2
        
        outcome = model.simulate_experiment(truemp, thisexp)
       
        updaterEXACT.update(outcome, thisexp)
        updaterABC.update(outcome, thisexp)
        
        if np.mod(4*idx_exp,n_exp)==0:
            fig = plt.figure()
            
            particles = updaterEXACT.particle_locations
            weights = updaterEXACT.particle_weights      
            particlesABC = updaterABC.particle_locations
            weightsABC = updaterABC.particle_weights
            
            #this is shameful hack to get the Kernal estimate
            temp = copy(updaterEXACT)
            temp.resample
            
            pdf = gaussian_kde(temp.particle_locations[:,0])
            
            x = np.linspace(0,1,1000)
            plt.plot(particles[:,0],weights, '.')
            plt.plot(x,pdf(x),'b')

            temp = copy(updaterABC)
            temp.resample
            
            pdf = gaussian_kde(temp.particle_locations[:,0])
            plt.plot(x,pdf(x),'r')

            plt.plot(particlesABC[:,0],weightsABC, '.r')
            plt.plot(truemp,0,'g')
            
    print "True param: {}".format(truemp)    
    print "Est. mean EXACT: {}".format(updaterEXACT.est_mean())
    print "Est. mean ABC: {}".format(updaterABC.est_mean())
    
    print "Error EXACT: {}".format(np.sum(np.abs(truemp[0]-updaterEXACT.est_mean())**2))
    print "Trace Cov EXACT: {}".format(np.trace(updaterEXACT.est_covariance_mtx()))
    print "Resample count EXACT: {}".format(updaterEXACT.resample_count)
    print "Error ABC: {}".format(np.sum(np.abs(truemp[0]-updaterABC.est_mean())**2))
    print "Trace Cov ABC: {}".format(np.trace(updaterABC.est_covariance_mtx()))
    print "Resample count ABC: {}".format(updaterABC.resample_count)

#    print "BCRB: {}".format(1/updater.current_bim)
#    print "Theoretical BCRB: {}".format(1/BIM)

        
    
    plt.show()  
