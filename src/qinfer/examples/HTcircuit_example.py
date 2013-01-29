#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# HTcircuit_example.py: HT-curcuit model module
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


## FEATURES ####################################################################

from __future__ import division

## IMPORTS #####################################################################

import numpy as np
import matplotlib.pyplot as plt
import time
import numpy.linalg as la
from scipy.stats.kde import gaussian_kde
from copy import copy

## Imports from within QInfer. ##
from .. import tomography, smc
from ..distributions import UniformDistribution
from ..resamplers import LiuWestResampler, ClusteringResampler

## External libraries bundled with QInfer. ##
from .._lib import docopt

## DOCUMENTATION ###############################################################

USAGE = """
Usage: HTcircuit_example.py [options]

-h, --help                  Prints this help and returns.
-m NH, --n_Hadamarded=NH    Specifies the number of qubits which get a Hadamard
                            gate applied before the controlled Unitary
-M NQ, --n_qubits=NQ        The total number of non-control qubits 
-n NP, --n_particles=NP     Specifies how many particles to use in the SMC
                            approximation. [default: 5000]
-e NE, --n_exp=NE           Specifies how many measurements are to be made.
                            [default: 100]
-a ALGO, --algorithm=ALGO   Specifies which algorithm to use; currently 'SMC'
                            and 'SMC-ABC' are supported. [default: SMC]
-r ALGO, --resampler=ALGO   Specifies which resampling algorithm to use;
                            currently 'LW', 'DBSCAN-LW' and 'WDBSCAN-LW' are
                            supported. [default: LW]
--lw-a=A                    Parameter ``a`` of the LW resampling algorithm.
                            [default: 0.98]
--dbscan-eps=EPS            Epsilon parameter for the DBSCAN-based resamplers.
                            [default: 0.5]
--dbscan-minparticles=N     Minimum number of particles allowed in a cluster by
                            the DBSCAN-based resamplers. [default: 5]
--wdbscan-pow=POW           Power by which the weight is to be raised in the
                            WDBSCAN weighting step. [default: 0.5]
--abctol=TOL                Specifies the tolerance used by the SMC-ABC
                            algorithm. [default: 8e-6]
--abcsim=SIM                Specifies how many simulations are used by each ABC
                            step. [default: 10000]
-v, --verbose               Prints additional debugging information.
"""

## TODO ########################################################################

"""
    - Add plotting options to USAGE.
    - Add printing options to USAGE.    
"""

## SCRIPT ######################################################################

if __name__ == "__main__":

    # Handle command-line arguments using docopt.
    args = docopt.docopt(USAGE)
    N_PARTICLES   = int(args['--n_particles'])
    N_HADAMARDED  = int(args['--n_Hadamarded'])
    N_QUBITS      = int(args['--n_qubits'])
    
    n_exp       = int(args['--n_exp'])
    algo        = args['--algorithm']
    resamp_algo = args['--resampler']
    abctol      = float(args['--abctol'])
    abcsim      = int(args['--abcsim'])
    verbose     = bool(args['--verbose'])
    lw_a        = float(args['--lw-a'])
    dbscan_eps  = float(args['--dbscan-eps'])
    dbscan_min  = float(args['--dbscan-minparticles'])
    wdbscan_pow = float(args['--wdbscan-pow'])
    
            
    # Model and prior initialization
    prior = UniformDistribution([-1,1])
    model = tomography.HTCircuitModel()

    f = np.arange(2**N_QUBITS)

    # a random invertible function    
    np.random.shuffle(f)
    
    expparams = {'nqubits':N_HADAMARDED,'boolf':f} 
    
    
    # Resampler initialization
    lw_args = {"a": lw_a}
    dbscan_args = {"eps": dbscan_eps, "min_particles": dbscan_min, "w_pow": wdbscan_pow}
    
    if resamp_algo == 'LW':
        resampler = LiuWestResampler(**lw_args)
    elif resamp_algo == 'DBSCAN-LW':
        resampler = ClusteringResampler(secondary_resampler=LiuWestResampler(**lw_args), weighted=False, quiet=not verbose, **dbscan_args)
    elif resamp_algo == 'WDBSCAN-LW':
        print "[WARN] The WDBSCAN-LW resampling algorithm is currently experimental, and may not work properly."
        resampler = ClusteringResampler(secondary_resampler=LiuWestResampler(), weighted=True, quiet=not verbose, **dbscan_args)
    else:
        raise ValueError('Must specify a valid resampler.')
        
    # SMC initialization
    if algo == 'SMC':
        use_like = True
        updater = smc.SMCUpdater(model, N_PARTICLES, prior, resampler=resampler)
    elif algo == 'SMC-ABC':
        use_like = False
        updater = smc.SMCUpdaterABC(model, N_PARTICLES, prior, resampler=resampler, abc_tol=abctol, abc_sim=abcsim)
    else:
        raise ValueError('Must specify a valid algorithm.')    
    
    tic = toc = None
    
    # Sample true set of modelparams
    truemp = prior.sample() 
    
    
    #Plotting intialization
    res = 1000
    p = np.linspace(-1,1,res)    
    L = np.ones((n_exp+1, res))       
    
    
    # Get all Bayesian up in here
    tic = time.time()
    for idx_exp in xrange(n_exp):
        outcome = model.simulate_experiment(truemp, expparams,use_like=use_like)        
        updater.update(outcome, expparams)
        
        temp = model.likelihood(np.array([outcome]),p,expparams)
        
        L[idx_exp+1,:] = L[idx_exp,:]*temp        
        norm = 2*np.sum(L[idx_exp+1,:])/res
        L[idx_exp+1,:] = L[idx_exp+1,:]/norm
        
        if np.mod(2*idx_exp,n_exp)==0:
            particles = updater.particle_locations
            weights = updater.particle_weights      
            
            Lm = L[idx_exp+1,:]
            fig = plt.figure()
            plt.plot(p,Lm, c = 'black')
            bme = 2*np.sum(p * Lm)/res
            plt.axvline(bme, c = 'blue', linewidth = 2)
            plt.axvline(truemp, c = 'red', linewidth = 2)            
            plt.axvline(updater.est_mean()[0], c = 'green', linewidth = 2)
            #plt.scatter(particles[:,0],np.zeros((N_PARTICLES,)),s = 50*(1+(weights-1/N_PARTICLES)*N_PARTICLES))
            temp = copy(updater)
            temp.resample()            
            pdf = gaussian_kde(temp.particle_locations[:,0])
            pdf.integrate_box_1d(-1,1)
            plt.plot(p,pdf(p),'g')
            plt.axis([-1,1,0,10])            
            
    est_mean = updater.est_mean()
    
    toc = time.time() - tic
        
    print "True param: {}".format(truemp)    
    print "Est. mean: {}".format(updater.est_mean())
    print "Est. cov: {}".format(updater.est_covariance_mtx())
    print "Error: {}".format(np.sum(np.abs(truemp[0]-updater.est_mean())**2))
    print "Trace Cov: {}".format(np.trace(updater.est_covariance_mtx()))
    print "Resample count: {}".format(updater.resample_count)
    print "Elapsed time: {}".format(toc)
 
        
    
    plt.show()  
