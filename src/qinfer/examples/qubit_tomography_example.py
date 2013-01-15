#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# SMC.py: Tomgraphic models module
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

## IMPORTS #####################################################################

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import time
from scipy.spatial import Delaunay
import numpy.linalg as la

## Imports from within QInfer. ##
from .. import tomography, smc
from ..utils import mvee, uniquify
from ..resamplers import LiuWestResampler, ClusteringResampler

## External libraries bundled with QInfer. ##
from .._lib import docopt

## DOCUMENTATION ###############################################################

USAGE = """
Usage: qubit_tomography_example.py [options]

-h, --help                  Prints this help and returns.
-n NP, --n_particles=NP     Specifies how many particles to use in the SMC
                            approximation. [default: 5000]
-e NE, --n_exp=NE           Specifies how many measurements are to be made.
                            [default: 100]
-a ALGO, --algorithm=ALGO   Specifies which algorithm to use; currently 'SMC'
                            and 'SMC-ABC' are supported. [default: SMC]
-r ALGO, --resampler=ALGO   Specifies which resampling algorithm to use;
                            currently 'LW', 'DBSCAN-LW' and 'WDBSCAN-LW' are
                            supported. [default: LW]
--abctol=TOL                Specifies the tolerance used by the SMC-ABC
                            algorithm. [default: 8e-6]
--abcsim=SIM                Specifies how many simulations are used by each ABC
                            step. [default: 10000]
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
    N_PARTICLES = int(args['--n_particles'])
    n_exp       = int(args['--n_exp'])
    algo        = args['--algorithm']
    resamp_algo = args['--resampler']
    abctol      = float(args['--abctol'])
    abcsim      = int(args['--abcsim'])
    
            
    # Model and prior initialization
    prior = tomography.HilbertSchmidtUniform()
    model = tomography.QubitStatePauliModel()
    expparams = np.array([
        ([1, 0, 0], 1), # Records are indicated by tuples.
        ([0, 1, 0], 1),
        ([0, 0, 1], 1)
    ], dtype=model.expparams_dtype)
    
    # Resampler initialization
    if resamp_algo == 'LW':
        resampler = LiuWestResampler()
    elif resamp_algo == 'DBSCAN-LW':
        resampler = ClusteringResampler(secondary_resampler=LiuWestResampler(), weighted=False)
    elif resamp_algo == 'WDBSCAN-LW':
        print "[WARN] The WDBSCAN-LW resampling algorithm is currently experimental, and may not work properly."
        resampler = ClusteringResampler(secondary_resampler=LiuWestResampler(), weighted=True)
    else:
        raise ValueError('Must specify a valid resampler.')
        
    # SMC initialization
    if algo == 'SMC':
        updater = smc.SMCUpdater(model, N_PARTICLES, prior, resampler=resampler)
    elif algo == 'SMC-ABC':
        updater = smc.SMCUpdaterABC(model, N_PARTICLES, prior, resampler=resampler, abc_tol=abctol, abc_sim=abcsim)
    else:
        raise ValueError('Must specify a valid algorithm.')
    
    
    tic = toc = None
    
    # Sample true set of modelparams
    truemp = np.array([prior.sample()])
    

    # Plot true state and prior
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    particles = updater.particle_locations
    
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray")

    ax.scatter(particles[:,0],particles[:,1],particles[:,2], s = 10)
    ax.scatter(truemp[:,0],truemp[:,1],truemp[:,2],c = 'red',s = 25)
 
    
    # Get all Bayesian up in here
    
    tic = time.time()
    for idx_exp in xrange(n_exp):
        # Randomly choose one of the three experiments from expparams and make
        # an array containing just that experiment.
        thisexp = expparams[np.newaxis, np.random.randint(0,3)]
        assert thisexp.shape == (1,), "Shape of thisexp is wrong--- that should never happen."
        
        outcome = model.simulate_experiment(truemp, thisexp)
       
        updater.update(outcome, thisexp)
        
        if np.mod(2*idx_exp,n_exp)==0:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            particles = updater.particle_locations
            weights = updater.particle_weights      
            maxweight = np.max(weights)

#            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
#            x=np.cos(u)*np.sin(v)
#            y=np.sin(u)*np.sin(v)
#            z=np.cos(v)
#
#            ax.plot_wireframe(x, y, z, color="gray")

            ax.scatter(particles[:,0],particles[:,1],particles[:,2], s = 10*(1+(weights-1/N_PARTICLES)*N_PARTICLES))
            ax.scatter(truemp[:,0],truemp[:,1],truemp[:,2],c = 'red', s= 25)
#            ax.scatter(thisexp[0,0]*(-1)**(outcome[0]),thisexp[0,1]*(-1)**(outcome[0]),thisexp[0,2]*(-1)**(outcome[0]),s = 50, c = 'black')
            
    est_mean = updater.est_mean()
    ax.scatter(est_mean[0],est_mean[1],est_mean[2],c = 'cyan', s = 25)    
    
    faces, vertices = updater.region_est_hull()
    
    items = Poly3DCollection(faces, facecolors=[(0, 0, 0, 0.1)])
    ax.add_collection(items)
    
    
    A, centroid = updater.region_est_ellipsoid(tol=0.0001)
    
    #PLot covariance ellipse
    U, D, V = la.svd(A)
    
    
    rx, ry, rz = [1/np.sqrt(d) for d in D]
    u, v = np.mgrid[0:2*np.pi:20j,-np.pi/2:np.pi/2:10j]    
    
    x=rx*np.cos(u)*np.cos(v)
    y=ry*np.sin(u)*np.cos(v)
    z=rz*np.sin(v)
        
    for idx in xrange(x.shape[0]):
        for idy in xrange(y.shape[1]):
            x[idx,idy],y[idx,idy],z[idx,idy] = np.dot(np.transpose(V),np.array([x[idx,idy],y[idx,idy],z[idx,idy]])) + centroid
            
            
    ax.plot_surface(x, y, z, cstride = 1, rstride = 1, alpha = 0.1)
    
#    #PLot covariance ellipse
#    U, D, V = la.svd(updater.est_covariance_mtx())
#    
#    rx, ry, rz = [np.sqrt(d)/0.5 for d in D]
#    center = updater.est_mean()    
#    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]    
#    x=rx*np.cos(u)*np.sin(v)
#    y=ry*np.sin(u)*np.sin(v)
#    z=rz*np.cos(v)
#    
#    
#    for idx in xrange(x.shape[0]):
#        for idy in xrange(x.shape[1]):
#            x[idx,idy],y[idx,idy],z[idx,idy] = np.dot(np.array([x[idx,idy],y[idx,idy],z[idx,idy]]),V) + center
#            
#            
#    ax.plot_surface(x, y, z, cstride = 1, rstride = 1, alpha = 0.1)
    
    
    toc = time.time() - tic
        
    print "True param: {}".format(truemp)    
    print "Est. mean: {}".format(updater.est_mean())
    print "Est. cov: {}".format(updater.est_covariance_mtx())
    print "Error: {}".format(np.sum(np.abs(truemp[0]-updater.est_mean())**2))
    print "Trace Cov: {}".format(np.trace(updater.est_covariance_mtx()))
    print "Resample count: {}".format(updater.resample_count)
    print "Elapsed time: {}".format(toc)
 
        
    
    plt.show()  
