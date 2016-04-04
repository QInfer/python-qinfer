#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# qubit_plotting_example.py: qubit tomography illustration module
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

## DOCUMENTATION ###############################################################

"""
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
--lw-a=A                    Parameter ``a`` of the LW resampling algorithm.
                            [default: 0.98]
--dbscan-eps=EPS            Epsilon parameter for the DBSCAN-based resamplers.
                            [default: 0.5]
--dbscan-minparticles=N     Minimum number of particles allowed in a cluster by
                            the DBSCAN-based resamplers. [default: 5]
--wdbscan-pow=POW           Power by which the weight is to be raised in the
                            WDBSCAN weighting step. [default: 0.5]
                            step. [default: 10000]
--no-plot                   Suppresses plotting when passed.
-v, --verbose               Prints additional debugging information.
"""

## FEATURES ####################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

## IMPORTS #####################################################################

from builtins import range

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import time
import numpy.linalg as la
import sys

## Imports from within QInfer. ##
from .. import tomography, smc
from ..resamplers import LiuWestResampler, ClusteringResampler
from ..distributions import HilbertSchmidtUniform

## External libraries bundled with QInfer. ##
from .._lib import docopt

## SCRIPT ######################################################################

if __name__ == "__main__":

    # Handle command-line arguments using docopt.
    args = docopt.docopt(__doc__, sys.argv[1:])
    
    N_PARTICLES = int(args['--n_particles'])
    n_exp       = int(args['--n_exp'])
    resamp_algo = args['--resampler']
    verbose     = bool(args['--verbose'])
    lw_a        = float(args['--lw-a'])
    dbscan_eps  = float(args['--dbscan-eps'])
    dbscan_min  = float(args['--dbscan-minparticles'])
    wdbscan_pow = float(args['--wdbscan-pow'])
    do_plot     = not bool(args['--no-plot'])
            
    # Model and prior initialization.
    prior = HilbertSchmidtUniform()
    model = tomography.QubitStatePauliModel()
    expparams = np.array([
        ([1, 0, 0], 1), # Records are indicated by tuples.
        ([0, 1, 0], 1),
        ([0, 0, 1], 1)
    ], dtype=model.expparams_dtype)
    
    # Resampler initialization.
    lw_args = {"a": lw_a}
    dbscan_args = {
        "eps": dbscan_eps,
        "min_particles": dbscan_min,
        "w_pow": wdbscan_pow
    }
    
    if resamp_algo == 'LW':
        resampler = LiuWestResampler(**lw_args)
    elif resamp_algo == 'DBSCAN-LW':
        resampler = ClusteringResampler(
            secondary_resampler=LiuWestResampler(**lw_args),
            weighted=False, quiet=not verbose, **dbscan_args
        )
    elif resamp_algo == 'WDBSCAN-LW':
        print("[WARN] The WDBSCAN-LW resampling algorithm is currently experimental, and may not work properly.")
        resampler = ClusteringResampler(
            secondary_resampler=LiuWestResampler(**lw_args),
            weighted=True, quiet=not verbose, **dbscan_args
        )
    else:
        raise ValueError('Must specify a valid resampler.')
        
    updater = smc.SMCUpdater(model, N_PARTICLES, prior, resampler=resampler)
    
    # Sample true set of modelparams
    truemp = np.array([prior.sample()])
    
    # Plot true state and prior
    if do_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        particles = updater.particle_locations
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')        
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color="gray", alpha = 0.5)
        ax.set_ylim(-1,1)
        ax.set_aspect('equal')

        ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2], s=10)
        ax.scatter(truemp[:, 0], truemp[:, 1], truemp[:, 2], c='red', s=25)
 
    # Record the start time.
    tic = time.time()
    
    # Get all Bayesian up in here.
    for idx_exp in range(n_exp):
        # Randomly choose one of the three experiments from expparams and make
        # an array containing just that experiment.
        thisexp = expparams[np.newaxis, np.random.randint(0, 3)]
        
        # Simulate an experiment according to the chosen expparams.
        outcome = model.simulate_experiment(truemp, thisexp)
       
        # Feed the data to the SMC particle updater.
        updater.update(outcome, thisexp)
        
        # Optionally plot the data so far.
        if do_plot and np.mod(4*idx_exp, n_exp) == 0:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')            
            particles = updater.particle_locations
            weights = updater.particle_weights      
            maxweight = np.max(weights)
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = np.cos(u) * np.sin(v)
            y = np.sin(u) * np.sin(v)
            z = np.cos(v)
            ax.plot_wireframe(x, y, z, color="gray", alpha = 0.5)
            ax.set_ylim(-1,1)
            ax.set_aspect('equal')
        
            ax.scatter(
                particles[:, 0], particles[:, 1], particles[:, 2],
                s=20 * (1 + (weights - 1 / N_PARTICLES) * N_PARTICLES)
            )
            temp = thisexp['axis'][0]*(-1)**outcome
            ax.scatter(temp[0], temp[1], temp[2], c='green', s=50)
            ax.scatter(truemp[:, 0], truemp[:, 1], truemp[:, 2], c='red', s=50)
            
    # Record how long it took us.
    toc = time.time() - tic
            
    # Print out summary statistics.    
    print("True param: {}".format(truemp))    
    print("Est. mean: {}".format(updater.est_mean()))
    print("Est. cov: {}".format(updater.est_covariance_mtx()))
    print("Error: {}".format(np.sum(np.abs(truemp[0]-updater.est_mean())**2)))
    print("Trace Cov: {}".format(np.trace(updater.est_covariance_mtx())))
    print("Resample count: {}".format(updater.resample_count))
    print("Elapsed time: {}".format(toc))
    

    est_mean = updater.est_mean()
    if do_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color="gray", alpha = 0.5)
        ax.set_ylim(-1,1)
        ax.set_aspect('equal')
        particles = updater.particle_locations
        weights = updater.particle_weights      
        maxweight = np.max(weights)

        ax.scatter(
            particles[:, 0], particles[:, 1], particles[:, 2],
            s=20 * (1 + (weights - 1 / N_PARTICLES) * N_PARTICLES)
        )
        
        #temp = thisexp['axis'][0]*(-1)**outcome
        #ax.scatter(temp[0], temp[1], temp[2], c='green', s=50)
        ax.scatter(truemp[:, 0], truemp[:, 1], truemp[:, 2], c='red', s=50)
        est_mean = updater.est_mean()
        ax.scatter(est_mean[0], est_mean[1], est_mean[2], c='cyan', s=25)    
        
        faces, vertices = updater.region_est_hull()
        
        items = Poly3DCollection(faces, facecolors=[(0, 0, 0, 0.1)])
        ax.add_collection(items)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color="gray", alpha = 0.5)
        ax.set_ylim(-1,1)
        ax.set_aspect('equal')

        particles = updater.particle_locations
        weights = updater.particle_weights      
        maxweight = np.max(weights)

        ax.scatter(
            particles[:, 0], particles[:, 1], particles[:, 2],
            s=20 * (1 + (weights - 1 / N_PARTICLES) * N_PARTICLES)
        )
        
        #temp = thisexp['axis'][0]*(-1)**outcome
        #ax.scatter(temp[0], temp[1], temp[2], c='green', s=50)
        ax.scatter(truemp[:, 0], truemp[:, 1], truemp[:, 2], c='red', s=50)
        est_mean = updater.est_mean()
        ax.scatter(est_mean[0], est_mean[1], est_mean[2], c='cyan', s=25)    
        
        faces, vertices = updater.region_est_hull()
        
        items = Poly3DCollection(faces, facecolors=[(0, 0, 0, 0.1)])
        ax.add_collection(items)

        A, centroid = updater.region_est_ellipsoid(tol=0.0001)

        # Plot MVEE ellipse.
        U, D, V = la.svd(A)
        
        
        rx, ry, rz = [1 / np.sqrt(d) for d in D]
        u, v = np.mgrid[0:(2 * np.pi):20j, -(np.pi / 2):(np.pi / 2):10j]

        
        x = rx * np.cos(u) * np.cos(v)
        y = ry * np.sin(u) * np.cos(v)
        z = rz * np.sin(v)
            
        for idx in range(x.shape[0]):
            for idy in range(y.shape[1]):
                x[idx, idy], y[idx, idy], z[idx, idy] = \
                    np.dot(
                        np.transpose(V),
                        np.array([x[idx,idy],y[idx,idy],z[idx,idy]])
                    ) + centroid
                
                
        ax.plot_surface(x, y, z, cstride = 1, rstride = 1, alpha = 0.1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color="gray", alpha = 0.5)
        ax.set_ylim(-1,1)
        ax.set_aspect('equal')
        particles = updater.particle_locations
        weights = updater.particle_weights      
        maxweight = np.max(weights)

        ax.scatter(
            particles[:, 0], particles[:, 1], particles[:, 2],
            s=20 * (1 + (weights - 1 / N_PARTICLES) * N_PARTICLES)
        )

        rx, ry, rz = [1 / np.sqrt(d) for d in D]
        u, v = np.mgrid[0:(2 * np.pi):20j, -(np.pi / 2):(np.pi / 2):10j]

        
        x = rx * np.cos(u) * np.cos(v)
        y = ry * np.sin(u) * np.cos(v)
        z = rz * np.sin(v)
            
        for idx in range(x.shape[0]):
            for idy in range(y.shape[1]):
                x[idx, idy], y[idx, idy], z[idx, idy] = \
                    np.dot(
                        np.transpose(V),
                        np.array([x[idx,idy],y[idx,idy],z[idx,idy]])
                    ) + centroid

        ax.scatter(truemp[:, 0], truemp[:, 1], truemp[:, 2], c='red', s=50)
        est_mean = updater.est_mean()
        ax.scatter(est_mean[0], est_mean[1], est_mean[2], c='cyan', s=25)                   
                
        ax.plot_surface(x, y, z, cstride = 1, rstride = 1, alpha = 0.1, linewidth = 0.1)

        A, centroid = la.inv(updater.est_covariance_mtx()),updater.est_mean()

        # Plot covariance ellipse.
        U, D, V = la.svd(A)

        
        rx, ry, rz = [np.sqrt(7.81/d) for d in D]
        u, v = np.mgrid[0:(2 * np.pi):20j, -(np.pi / 2):(np.pi / 2):10j]

        
        x = rx * np.cos(u) * np.cos(v)
        y = ry * np.sin(u) * np.cos(v)
        z = rz * np.sin(v)
            
        for idx in range(x.shape[0]):
            for idy in range(y.shape[1]):
                x[idx, idy], y[idx, idy], z[idx, idy] = \
                    np.dot(
                        np.transpose(V),
                        np.array([x[idx,idy],y[idx,idy],z[idx,idy]])
                    ) + centroid

        ax.plot_surface(x, y, z, cstride = 1, rstride = 1, alpha = 0.1, color = 'red', linewidth = 0.1)

        plt.show()
        
