#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# clustering.py: Wraps clustering algorithms provided by SciKit-Learn.
##
# © 2017, Chris Ferrie (csferrie@gmail.com) and
#         Christopher Granade (cgranade@cgranade.com).
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the copyright holder nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
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

from qinfer.utils import outer_product
from qinfer._exceptions import ResamplerWarning
import qinfer.metrics as metrics

try:
    import sklearn
    import sklearn.cluster
    import sklearn.metrics
    import sklearn.metrics.pairwise
except ImportError:
    try:
        import logging
        logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        logger.info("Could not import scikit-learn. Clustering support is disabled.")
    except:
        pass
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


