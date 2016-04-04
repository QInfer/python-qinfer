#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# clustering.py: Wraps clustering algorithms provided by SciKit-Learn.
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

## FEATURES ###################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

## ALL ########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
    'particle_clusters',
    'NOISE'
]

## IMPORTS ####################################################################

from builtins import range

import warnings

import numpy as np
import scipy.linalg as la

from qinfer.utils import outer_product, particle_meanfn, particle_covariance_mtx
from qinfer._exceptions import ResamplerWarning
import qinfer.metrics as metrics

try:
    import sklearn
    import sklearn.cluster
    import sklearn.metrics
    import sklearn.metrics.pairwise
except ImportError:
    warnings.warn("Could not import scikit-learn. Some features may not work.",
        ImportWarning)
    sklearn = None

## CONSTANTS ##################################################################

NOISE = -1

## FUNCTIONS ##################################################################

def particle_clusters(
        particle_locations, particle_weights=None,
        eps=0.5, min_particles=5, metric='euclidean',
        weighted=False, w_pow=0.5,
        quiet=True
    ):
    """
    Yields an iterator onto tuples ``(cluster_label, cluster_particles)``,
    where ``cluster_label`` is an `int` identifying the cluster (or ``NOISE``
    for the particles lying outside of all clusters), and where
    ``cluster_particles`` is an array of ``dtype`` `bool` specifying the indices
    of all particles in that cluster. That is, particle ``i`` is in the cluster
    if ``cluster_particles[i] == True``.
    """
    
    
    if weighted == True and particle_weights is None:
        raise ValueError("Weights must be specified for weighted clustering.")
        
    # Allocate new arrays to hold the weights and locations.        
    new_weights = np.empty(particle_weights.shape)
    new_locs    = np.empty(particle_locations.shape)
    
    # Calculate and possibly reweight the metric.
    if weighted:
        M = sklearn.metrics.pairwise.pairwise_distances(particle_locations, metric=metric)
        M = metrics.weighted_pairwise_distances(M, particle_weights, w_pow=w_pow)
    
        # Create and run a SciKit-Learn DBSCAN clusterer.
        clusterer = sklearn.cluster.DBSCAN(
            min_samples=min_particles,
            eps=eps,
            metric='precomputed'
        )
        cluster_labels = clusterer.fit_predict(M)
    else:
        clusterer = sklearn.cluster.DBSCAN(
            min_samples=min_particles,
            eps=eps,
            metric=metric
        )
        cluster_labels = clusterer.fit_predict(particle_locations)
    
    # Find out how many clusters were identified.
    # Cluster counting logic from:
    # [http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html].
    is_noise = -1 in cluster_labels
    n_clusters = len(set(cluster_labels)) - (1 if is_noise else 0)
    
    # If more than 10% of the particles were labeled as NOISE,
    # warn.
    n_noise = np.sum(cluster_labels == -1)
    if n_noise / particle_weights.shape[0] >= 0.1:
        warnings.warn("More than 10% of the particles were classified as NOISE. Consider increasing the neighborhood size ``eps``.", ResamplerWarning)
    
    # Print debugging info.
    if not quiet:
        print("[Clustering] DBSCAN identified {} cluster{}. "\
              "{} particles identified as NOISE.".format(
                  n_clusters, "s" if n_clusters > 1 else "", n_noise
              ))
        
    # Loop over clusters, calling the secondary resampler for each.
    # The loop should include -1 if noise was found.
    for idx_cluster in range(-1 if is_noise else 0, n_clusters):
        # Grab a boolean array identifying the particles in a  particular
        # cluster.
        this_cluster = cluster_labels == idx_cluster
        
        yield idx_cluster, this_cluster
    

