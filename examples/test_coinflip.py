#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_coinflip.py: Simple demonstration.
##
# © 2013 Chris Ferrie (csferrie@gmail.com) and
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

from abstract_model import Model
    
## CLASSES ##

class SimpleCoinModel(Model):
    
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
        return modelparams.all() >= 0 and modelparams.all() <= 1
    
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
        return Model.pr0_to_likelihood_array(outcomes, modelparams)

## TESTING CODE ################################################################

if __name__ == "__main__":
    from distributions import UniformDistribution
    import smc
    import matplotlib.pyplot as plt
    from scipy.stats.kde import gaussian_kde
    from copy import copy
#============ BAYES ============================================
#    model = SimpleCoinModel()
#
#    res = 1000
#    p = np.linspace(0,1,res)
#    
#    n_exp = 10
#    
#    L = np.zeros((n_exp, res))
#            
#    outcomes = np.array([[0],[0],[1],[0],[1],[0],[0],[0],[1],[0]])
#    
#    for idx_exp in range(n_exp):
#        thisexp = np.array([np.random.random()],dtype=model.expparams_dtype)
#        outcome = outcomes[idx_exp]
#        temp = np.log(model.likelihood(outcome,p,thisexp))
#        L[idx_exp,:] = temp        
#        Ls = np.cumsum(L,0)
#        if (idx_exp) % 1 == 0:
#            Lm = np.exp(Ls[idx_exp,:])
#            norm = np.sum(Lm)/res
#            fig = plt.figure()
#            plt.plot(p,Lm/norm, c = 'black')
#
#            mle = p[np.argmax(Lm)]
#            bme = np.sum(p * Lm) / norm / res
#            plt.axvline(mle,linewidth = 2)
#            plt.axvline(bme, c = 'red', linewidth = 2)
#            plt.axis([0,1,0,3])
    


#============ SMC ============================================== 
    N_PARTICLES = 10
    
    prior = UniformDistribution([0,1])
    model = SimpleCoinModel()
    updater = smc.SMCUpdater(model, N_PARTICLES, prior)

    res = 100
    p = np.linspace(0,1,res)
    
    n_exp = 10
    
    L = np.zeros((n_exp, res))
    
    outcomes = np.array([[0],[0],[1],[0],[1],[0],[0],[0],[1],[0]])
    
    for idx_exp in range(n_exp):
        thisexp = np.array([np.random.random()],dtype=model.expparams_dtype)
        outcome = outcomes[idx_exp]        
        temp = np.log(model.likelihood(outcome,p,thisexp))
        L[idx_exp,:] = temp        
        Ls = np.cumsum(L,0)
        updater.update(outcome,thisexp)
        if (idx_exp) % 1 == 0:
            particles = updater.particle_locations
            weights = updater.particle_weights
            Lm = np.exp(Ls[idx_exp,:])
            norm = np.sum(Lm)/res
            fig = plt.figure()
            plt.plot(p,Lm/norm, c = 'black')
            bme = np.sum(p * Lm) / norm / res
            plt.axvline(bme, c = 'red', linewidth = 2)
            print updater.est_mean()[0]
            plt.axvline(updater.est_mean()[0], linestyle = '--', c = 'blue', linewidth = 2)
            plt.scatter(particles[:,0],np.zeros((N_PARTICLES,)),s = 50*(1+(weights-1/N_PARTICLES)*N_PARTICLES))
#            temp = copy(updater)
#            temp.resample
#            
#            pdf = gaussian_kde(temp.particle_locations[:,0])
#            plt.plot(p,pdf(p),'b')
            plt.axis([0,1,0,3])

    plt.show()

