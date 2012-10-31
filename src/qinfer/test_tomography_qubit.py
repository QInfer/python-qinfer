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
import tomography, smc
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import time
from scipy.spatial import Delaunay

if __name__ == "__main__":

    N_PARTICLES = 250
            
    # Model and prior initialization
    prior = tomography.HilbertSchmidtUniform()
    model = tomography.QubitStatePauliModel()
    expparams = np.array([
        ([1, 0, 0], 1), # Records are indicated by tuples.
        ([0, 1, 0], 1),
        ([0, 0, 1], 1)
    ], dtype=model.expparams_dtype)
    
    # SMC initialization
    updater = smc.SMCUpdater(model, N_PARTICLES, prior,resample_a=.98, resample_thresh=0.5)
    
    
    tic = toc = None
    
    # Sample true set of modelparams
    truemp = np.array([prior.sample()])
    

    # Plot true state and prior
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    particles = updater.particle_locations
    
    ax.scatter(particles[:,0],particles[:,1],particles[:,2], s = 10)
    ax.scatter(truemp[:,0],truemp[:,1],truemp[:,2],c = 'red',s = 25)
 
    
    # Get all Bayesian up in here
    n_exp = 100
    tic = time.time()
    for idx_exp in xrange(n_exp):
        # Randomly choose one of the three experiments from expparams and make
        # an array containing just that experiment.
        thisexp = expparams[np.newaxis, np.random.randint(0,3)]
        assert thisexp.shape == (1,), "Shape of thisexp is wrong--- that should never happen."
        
        outcome = model.simulate_experiment(truemp, thisexp)
       
        updater.update(outcome, thisexp)
        
        if np.mod(10*idx_exp,n_exp)==0:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            particles = updater.particle_locations
            weights = updater.particle_weights      
            maxweight = np.max(weights)
       
            ax.scatter(particles[:,0],particles[:,1],particles[:,2], s = 10*(1+(weights-1/N_PARTICLES)*N_PARTICLES))
            ax.scatter(truemp[:,0],truemp[:,1],truemp[:,2],c = 'red', s= 25)
#            ax.scatter(thisexp[0,0]*(-1)**(outcome[0]),thisexp[0,1]*(-1)**(outcome[0]),thisexp[0,2]*(-1)**(outcome[0]),s = 50, c = 'black')

    est_mean = updater.est_mean()
    ax.scatter(est_mean[0],est_mean[1],est_mean[2],c = 'cyan', s = 25)    
    
    points = updater.est_credible_region(level = .99)
    tri = Delaunay(points)
    faces = []
    for ia, ib, ic in tri.convex_hull:
        faces.append(points[[ia, ib, ic]])    
    
    items = Poly3DCollection(faces, facecolors=[(0, 0, 0, 0.1)])
    ax.add_collection(items)

    toc = time.time() - tic
        
    print "True param: {}".format(truemp)    
    print "Est. mean: {}".format(updater.est_mean())
    print "Est. cov: {}".format(updater.est_covariance_mtx())
    print "Error: {}".format(np.sum(np.abs(truemp[0]-updater.est_mean())**2))
    print "Trace Cov: {}".format(np.trace(updater.est_covariance_mtx()))
    print "Resample count: {}".format(updater.resample_count)
    print "Elapsed time: {}".format(toc)
 
        
    
    plt.show()  
