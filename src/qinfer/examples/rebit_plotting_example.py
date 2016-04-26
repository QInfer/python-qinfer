#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# rebit_plotting_example.py: rebit tomography illustration module
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

## DOCUMENTATION ###############################################################

"""
Usage: rebit_tomography_example.py [options]

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
import matplotlib.pyplot as plt

import time

import numpy.linalg as la
from scipy.spatial import Delaunay

import sys

## Imports from within QInfer. ##
from .. import tomography, smc
from ..resamplers import LiuWestResampler, ClusteringResampler
from ..utils import mvee, uniquify

## External libraries bundled with QInfer. ##
from .._lib import docopt

## CLASSES #####################################################################

class HilbertSchmidtUniform(object):
    """
    Creates a new Hilber-Schmidt uniform prior on state space of dimension ``dim``.
    See e.g. [Mez06]_ and [Mis12]_.
    """
    def __init__(self):
        self.dim = 2
        
    def sample(self):
        #Generate random unitary (see e.g. http://arxiv.org/abs/math-ph/0609050v2)        
        g = (np.random.randn(self.dim,self.dim) + 1j*np.random.randn(self.dim,self.dim))/np.sqrt(2.0)
        q,r = la.qr(g)
        d = np.diag(r)
        
        ph = d/np.abs(d)
        ph = np.diag(ph)
        
        U = np.dot(q,ph)

        #Generate random matrix        
        z = np.random.randn(self.dim,self.dim) + 1j*np.random.randn(self.dim,self.dim)
        
        rho = np.dot(np.dot(np.identity(self.dim)+U,np.dot(z,z.conj().transpose())),np.identity(self.dim)+U.conj().transpose())
        rho = rho/np.trace(rho)
        
        # TODO: generalize to Heisenberg-Weyl groups.
        y = np.real(np.trace(np.dot(rho,np.array([[0,-1j],[1j,0]]))))
        x = np.real(np.trace(np.dot(rho,np.array([[0,1],[1,0]]))))
        
        return np.array([x,y])

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
    model = tomography.RebitStatePauliModel()
    expparams = np.array([
        ([1, 0], 1), # Records are indicated by tuples.
        ([0, 1], 1)
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
        ax = plt.gca()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_ylim(-1,1)
        ax.set_xlim(-1,1)
        ax.set_aspect('equal')
        
        u = np.linspace(0,2*np.pi,100)
        x = np.cos(u)
        y = np.sin(u)
        
        plt.plot(x,y)

        particles = updater.particle_locations
     
        plt.scatter(particles[:, 0], particles[:, 1], s=10)
        plt.scatter(truemp[:, 0], truemp[:, 1], c='red', s=50)
        est_mean = updater.est_mean()
        plt.scatter(est_mean[0], est_mean[1], c='cyan', s=50)
 
    # Record the start time.
    tic = time.time()
    
    # Get all Bayesian up in here.
    for idx_exp in range(n_exp):
        # Randomly choose one of the three experiments from expparams and make
        # an array containing just that experiment.
        thisexp = expparams[np.newaxis, np.random.randint(0, 2)]
        
        # Simulate an experiment according to the chosen expparams.
        outcome = model.simulate_experiment(truemp, thisexp)
       
        # Feed the data to the SMC particle updater.
        updater.update(outcome, thisexp)
            
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
        ax = plt.gca()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_ylim(-1,1)
        ax.set_xlim(-1,1)
        ax.set_aspect('equal')

        particles = updater.particle_locations
        weights = updater.particle_weights      
        maxweight = np.max(weights)

        u = np.linspace(0,2*np.pi,100)
        x = np.cos(u)
        y = np.sin(u)
    
        plt.plot(x,y) 
    
        plt.scatter(
            particles[:, 0], particles[:, 1],
            s=20 * (1 + (weights - 1 / N_PARTICLES) * N_PARTICLES)
        )
        temp = thisexp['axis'][0]*(-1)**outcome
        #plt.scatter(temp[0], temp[1], c='green', s=50)
        plt.scatter(truemp[:, 0], truemp[:, 1], c='red', s=50)
        plt.scatter(est_mean[0], est_mean[1], c='cyan', s=50)    
        
        points = updater.est_credible_region(level = 0.95)
        tri = Delaunay(points)
        faces = []
        hull = tri.convex_hull
        
        for ia, ib in hull:
            faces.append(points[[ia, ib]])    

        vertices = points[uniquify(hull.flatten())]
        temp = vertices - np.mean(vertices, 0)
        
        idx_srt = np.argsort(np.arctan2(temp[:, 1], temp[:, 0]))
        idx_srt = np.append(idx_srt,idx_srt[0])     
        
        fig = plt.figure()
        ax = plt.gca()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_ylim(-1,1)
        ax.set_xlim(-1,1)
        ax.set_aspect('equal')
        particles = updater.particle_locations
        weights = updater.particle_weights      
        maxweight = np.max(weights)

        u = np.linspace(0,2*np.pi,100)
        x = np.cos(u)
        y = np.sin(u)
        
        plt.plot(x,y)

        plt.scatter(
            particles[:, 0], particles[:, 1],
            s=20 * (1 + (weights - 1 / N_PARTICLES) * N_PARTICLES)
        )
        
        plt.scatter(truemp[:, 0], truemp[:, 1], c='red', s=50)
        plt.scatter(est_mean[0], est_mean[1], c='cyan', s=25) 
        
        
        x = vertices[:,0][idx_srt]
        y = vertices[:,1][idx_srt]
        
        plt.plot(x,y)
        
                   
        fig = plt.figure()
        ax = plt.gca()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_ylim(-1,1)
        ax.set_xlim(-1,1)
        ax.set_aspect('equal')

        particles = updater.particle_locations
        weights = updater.particle_weights      
        maxweight = np.max(weights)

        u = np.linspace(0,2*np.pi,100)
        x = np.cos(u)
        y = np.sin(u)
        
        plt.plot(x,y)

        plt.scatter(
            particles[:, 0], particles[:, 1],
            s=20 * (1 + (weights - 1 / N_PARTICLES) * N_PARTICLES)
        )
        
        plt.scatter(truemp[:, 0], truemp[:, 1], c='red', s=50)
        plt.scatter(est_mean[0], est_mean[1], c='cyan', s=25) 
        
        
        x = vertices[:,0][idx_srt]
        y = vertices[:,1][idx_srt]
        
        plt.plot(x,y)        

        A, centroid = mvee(vertices, 0.001)

        # Plot mvee ellipse.
        U, D, V = la.svd(A)
        
        
        rx, ry = [1 / np.sqrt(d) for d in D]
        u = np.linspace(0,(2 * np.pi),100)

        
        x = rx * np.cos(u)
        y = ry * np.sin(u)

        for idx in range(x.shape[0]):
                x[idx], y[idx] = \
                    np.dot(
                        np.transpose(V),
                        np.array([x[idx],y[idx]])
                    ) + centroid
        
        plt.plot(x,y)
        
        # Plot covariance ellipse.
        U, D, V = la.svd(la.inv(updater.est_covariance_mtx()))
        
        
        rx, ry = [np.sqrt(6/d) for d in D]
        u = np.linspace(0,(2 * np.pi),100)

        
        x = rx * np.cos(u)
        y = ry * np.sin(u)

        for idx in range(x.shape[0]):
                x[idx], y[idx] = \
                    np.dot(
                        np.transpose(V),
                        np.array([x[idx],y[idx]])
                    ) + updater.est_mean()
        
        plt.plot(x,y)

        plt.show()
        
