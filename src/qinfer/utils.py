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

## FEATURES ###################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

## IMPORTS ####################################################################

from builtins import range

import warnings

import numpy as np
import numpy.linalg as la

from scipy.stats import logistic, binom
from scipy.special import gammaln, gamma
from scipy.linalg import sqrtm

from qinfer._exceptions import ApproximationWarning

## FUNCTIONS ##################################################################

def binomial_pdf(N,n,p):
    return binom(N, p).pmf(n)

def outer_product(vec):        
    return (
        np.dot(vec[:, np.newaxis], vec[np.newaxis, :])
        if len(vec.shape) == 1 else
        np.dot(vec, vec.T)
        )
        
def particle_meanfn(weights, locations, fn=None):
    fn_vals = fn(locations) if fn is not None else locations
    return np.sum(weights * fn_vals.transpose([1, 0]),
        axis=1)

    
def particle_covariance_mtx(weights,locations):
    """
    Returns an estimate of the covariance of a distribution
    represented by a given set of SMC particle.
        
    :param weights: An array containing the weights of each
        particle.
    :param location: An array containing the locations of
        each particle.
    :rtype: :class:`numpy.ndarray`, shape
        ``(n_modelparams, n_modelparams)``.
    :returns: An array containing the estimated covariance matrix.
    """
    # TODO: add shapes to docstring.        
        
    # Find the mean model vector, shape (n_modelparams, ).
    mu = particle_meanfn(weights, locations)
    
    # Transpose the particle locations to have shape
    # (n_modelparams, n_particles).
    xs = locations.transpose([1, 0])
    # Give a shorter name to the particle weights, shape (n_particles, ).
    ws = weights

    cov = (
        # This sum is a reduction over the particle index, chosen to be
        # axis=2. Thus, the sum represents an expectation value over the
        # outer product $x . x^T$.
        #
        # All three factors have the particle index as the rightmost
        # index, axis=2. Using the Einstein summation convention (ESC),
        # we can reduce over the particle index easily while leaving
        # the model parameter index to vary between the two factors
        # of xs.
        #
        # This corresponds to evaluating A_{m,n} = w_{i} x_{m,i} x_{n,i}
        # using the ESC, where A_{m,n} is the temporary array created.
        np.einsum('i,mi,ni', ws, xs, xs)
        # We finish by subracting from the above expectation value
        # the outer product $mu . mu^T$.
        - np.dot(mu[..., np.newaxis], mu[np.newaxis, ...])
    )
    
    # The SMC approximation is not guaranteed to produce a
    # positive-semidefinite covariance matrix. If a negative eigenvalue
    # is produced, we should warn the caller of this.
    assert np.all(np.isfinite(cov))
    if not np.all(la.eig(cov)[0] >= 0):
        warnings.warn('Numerical error in covariance estimation causing positive semidefinite violation.', ApproximationWarning)

    return cov            


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

def format_uncertainty(value, uncertianty, scinotn_break=4):
    if uncertianty == 0:
        # Return the exact number, without the ± annotation as a fixed point
        # number, since all digits matter.
        # FIXME: this assumes a precision of 6; need to select that dynamically.
        return "{0:f}".format(value)
    else:
        # Return a string of the form "0.00 \pm 0.01".
        mag_unc = int(np.log10(np.abs(uncertianty)))
        # Zero should be printed as a single digit; that is, as wide as str "1".
        mag_val = int(np.log10(np.abs(value))) if value != 0 else 0
        n_digits = max(mag_val - mag_unc, 0)
            

        if abs(mag_val) < abs(mag_unc) and abs(mag_unc) > scinotn_break:
            # We're formatting something close to zero, so recale uncertianty
            # accordingly.
            scale = 10**mag_unc
            return r"({{0:0.{0}f}} \pm {{1:0.{0}f}}) \times 10^{{2}}".format(
                n_digits
            ).format(
                value / scale,
                uncertianty / scale,
                mag_unc
           )
        if abs(mag_val) <= scinotn_break:
            return r"{{0:0.{n_digits}f}} \pm {{1:0.{n_digits}f}}".format(n_digits=n_digits).format(value, uncertianty)
        else:
            scale = 10**mag_val
            return r"({{0:0.{0}f}} \pm {{1:0.{0}f}}) \times 10^{{2}}".format(
                n_digits
            ).format(
                value / scale,
                uncertianty / scale,
                mag_val
           )
           
def compactspace(scale, n):
    r"""
    Returns points :math:`x` spaced in the open interval
    :math:`(-\infty, \infty)`  by linearly spacing in the compactified
    coordinate :math:`s(x) = e^{-\alpha x} / (1 + e^{-\alpha x})^2`,
    where :math:`\alpha` is a scale factor.
    """
    logit = logistic(scale=scale).ppf
    compact_xs = np.linspace(0, 1, n + 2)[1:-1]
    return logit(compact_xs)
           
    
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
    print(A)
    
    # point it and some other stuff
    U, D, V = la.svd(A)    
        
    rx, ry, rz = [1/np.sqrt(d) for d in D]
    u, v = np.mgrid[0:2*np.pi:20j,-np.pi/2:np.pi/2:10j]    
    
    x=rx*np.cos(u)*np.cos(v)
    y=ry*np.sin(u)*np.cos(v)
    z=rz*np.sin(v)
            
    for idx in range(x.shape[0]):
        for idy in range(y.shape[1]):
            x[idx,idy],y[idx,idy],z[idx,idy] = np.dot(np.transpose(V),np.array([x[idx,idy],y[idx,idy],z[idx,idy]])) + centroid
            
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0],points[:,1],points[:,2])    
    ax.plot_surface(x, y, z, cstride = 1, rstride = 1, alpha = 0.1)
    plt.show()
 

def pretty_time(secs, force_h=False, force_m=False):
    if secs > 86400:
        return "{d} days, ".format(d=int(secs//86400)) + pretty_time(secs % 86400, force_h=True)
    elif force_h or secs > 3600:
        return "{h}:".format(h=int(secs//3600)) + pretty_time(secs % 3600, force_m=True)
    elif force_m or secs > 60:
        return (
            "{m:0>2}:{s:0>2}" if force_m else "{m}:{s:0>2}"
        ).format(m=int(secs//60), s=int(secs%60))
    else:
        return "{0:0.2f} seconds".format(secs)
