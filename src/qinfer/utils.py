#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# utils.py : some auxiliary functions
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

## IMPORTS ####################################################################

from __future__ import division
import numpy as np
from scipy.special import gammaln, gamma

from scipy.linalg import sqrtm

import numpy.linalg as la


###############################################################################

#TODO: cases for p=0 or p=1
def binomial_pdf(N,n,p):
    logprob = gammaln(N+1)-gammaln(n+1)- gammaln(N-n+1)  \
        + n*np.log(p)+(N-n)*np.log(1-p)
    return np.exp(logprob)

def outer_product(vec):        
    return (
        np.dot(vec[:, np.newaxis], vec[np.newaxis, :])
        if len(vec.shape) == 1 else
        np.dot(vec, vec.T)
        )
        
def particle_meanfn(weights, locations, fn):
    fn_vals = fn(locations)
    return np.sum(weights * fn_vals.transpose([1, 0]),
        axis=1)

    
def particle_covariance_mtx(weights,locations):
        
        xs = locations.transpose([1, 0])
        ws = weights
        
        mu = np.sum(ws * xs, axis = 1)
        
        return (
            np.sum(
                ws * xs[:, np.newaxis, :] * xs[np.newaxis, :, :],
                axis=2
                )
            ) - np.dot(mu[..., np.newaxis], mu[np.newaxis, ...])
            


def ellipsoid_volume(A=None, invA=None):
    
    if invA is None and A is None:
        raise ValueError("Must pass either inverse(A) or A.")
        
    if invA is None and A is not None:
        invA = la.inv(A)
    
    # Find the unit sphere volume.
    # http://en.wikipedia.org/wiki/Unit_sphere#General_area_and_volume_formulas
    n  = invA.shape[0]
    Vn = (np.pi ** (n/2)) / gamma(1 + (n/2))
    
    return Vn * la.det(sqrtm(invA))

def mvee(points,tol=0.001):
    N, d = points.shape
    
    Q = np.zeros([N,d+1])
    Q[:,0:d] = points[0:N,0:d]  
    Q[:,d] = np.ones([1,N])
    
    Q = np.transpose(Q)
    points = np.transpose(points)
    count = 1
    err = 1
    u = (1/N) * np.ones(shape = (N,))

    # Khachiyan Algorithm TODO:find ref
    while err > tol:
        
        X = np.dot(np.dot(Q, np.diag(u)), np.transpose(Q))
        M = np.diag( np.dot(np.dot(np.transpose(Q), la.inv(X)),Q)) 
        jdx = np.argmax(M)
        step_size = (M[jdx] - d - 1)/((d+1)*(M[jdx] - 1))
        new_u = (1 - step_size)*u 
        new_u[jdx] = new_u[jdx] + step_size
        count = count + 1
        err = la.norm(new_u - u)       
        u = new_u
    
    U = np.diag(u)    
    c = np.dot(points,u)
    A = (1/d) * la.inv(np.dot(np.dot(points,U), np.transpose(points)) - np.outer(c,c) )    
    return A, np.transpose(c)

def uniquify(seq):
    #from http://stackoverflow.com/a/480227/1205799
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if x not in seen and not seen_add(x)]
    
    
#==============================================================================
#Test Code
if __name__ == "__main__":

    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt
    from scipy.spatial import Delaunay
    
    #some random points
    points = np.array([[ 0.53135758, -0.25818091, -0.32382715], 
    [ 0.58368177, -0.3286576,  -0.23854156,], 
    [ 0.18741533,  0.03066228, -0.94294771], 
    [ 0.65685862, -0.09220681, -0.60347573],
    [ 0.63137604, -0.22978685, -0.27479238],
    [ 0.59683195, -0.15111101, -0.40536606],
    [ 0.68646128,  0.0046802,  -0.68407367],
    [ 0.62311759,  0.0101013,  -0.75863324]])
    
    # compute mvee
    A, centroid = mvee(points)
    print A
    
    # point it and some other stuff
    U, D, V = la.svd(A)    
        
    rx, ry, rz = [1/np.sqrt(d) for d in D]
    u, v = np.mgrid[0:2*np.pi:20j,-np.pi/2:np.pi/2:10j]    
    
    x=rx*np.cos(u)*np.cos(v)
    y=ry*np.sin(u)*np.cos(v)
    z=rz*np.sin(v)
            
    for idx in xrange(x.shape[0]):
        for idy in xrange(y.shape[1]):
            x[idx,idy],y[idx,idy],z[idx,idy] = np.dot(np.transpose(V),np.array([x[idx,idy],y[idx,idy],z[idx,idy]])) + centroid
            
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0],points[:,1],points[:,2])    
    ax.plot_surface(x, y, z, cstride = 1, rstride = 1, alpha = 0.1)
    plt.show()
 
