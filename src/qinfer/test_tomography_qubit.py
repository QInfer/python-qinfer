# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 18:53:24 2012

@author: csferrie
"""

## FEATURES ####################################################################

from __future__ import division

## IMPORTS #####################################################################

import getpass

import sys
sys.path.append({
    'csferrie': 'C:/Users/csferrie/Documents/GitHub/python-qinfer/src/',
    'cgranade': '/home/cgranade/academics/software-projects/python-qinfer/src/'
}[getpass.getuser()])

import numpy as np
from qinfer import tomography, smc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":

    N_PARTICLES = 250
            
    # Model and prior initialization
    prior = tomography.HaarUniform()
    model = tomography.QubitStatePauliModel()
    expparams = np.array([1])
    
    # SMC initialization
    updater = smc.SMCUpdater(model, N_PARTICLES, prior,resample_a=0.98, resample_thresh=0)
    
    
    tic = toc = None
    
    # Sample true set of modelparams
    truemp = np.array([prior.sample()])
    
    # Get all Bayesian up in here
    n_exp = 2
    tic = time.time()
    for idx_exp in xrange(n_exp):
        
        outcome = model.simulate_experiment(truemp, expparams)
        updater.update(outcome, expparams)
        
    toc = time.time() - tic
        
    print "True param: {}".format(truemp)    
    print "Est. mean: {}".format(updater.est_mean())
    print "Est. cov: {}".format(updater.est_covariance_mtx())
    print "Error: {}".format(np.sum(np.abs(truemp[0]-updater.est_mean())**2))
    print "Trace Cov: {}".format(np.trace(updater.est_covariance_mtx()))
    print "Resample count: {}".format(updater.resample_count)
    print "Elapsed time: {}".format(toc)
 
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
        
    particles = updater.particle_locations
    weights = updater.particle_weights
    minweight = min(weights)
    maxweight = max(weights)
   
    ax.scatter(particles[:,0],particles[:,1],particles[:,2], s = 100*weights/maxweight)
    ax.scatter(truemp[:,0],truemp[:,1],truemp[:,2],c = 'red')
    
    plt.show()  
