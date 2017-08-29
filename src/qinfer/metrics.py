#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# metrics.py: Metrics for use with SciPy and sklearn functions.
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
from __future__ import division

## ALL ########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
    'weighted_pairwise_distances'
]

## IMPORTS ####################################################################

import numpy as np
import scipy.linalg as la
import warnings

from qinfer.utils import outer_product

try:
    import sklearn
    import sklearn.metrics
    import sklearn.metrics.pairwise
except ImportError:
    try:
        import logging
        logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        logger.info("Could not import scikit-learn. Clustering metrics are disabled.")
    except:
        pass
    sklearn = None

## FUNCTIONS ##################################################################

def rescaled_distance_mtx(p, q):
    r"""
    Given two particle updaters for the same model, returns a matrix
    :math:`\matr{d}` with elements

    .. math::
        \matr{d}_{i,j} = \left\Vert \sqrt{\matr{Q}} \cdot
            (\vec{x}_{p, i} - \vec{x}_{q, j}) \right\Vert_2,

    where :math:`\matr{Q}` is the scale matrix of the model,
    :math:`\vec{x}_{p,i}` is the :math:`i`th particle of ``p``, and where
    :math:`\vec{x}_{q,i}` is the :math:`i`th particle of ``q`.

    :param qinfer.smc.SMCUpdater p: SMC updater for the distribution
        :math:`p(\vec{x})`.
    :param qinfer.smc.SMCUpdater q: SMC updater for the distribution
        :math:`q(\vec{x})`.

    Either or both of ``p`` or ``q`` can simply be the locations array for
    an :ref:`SMCUpdater`.
    """

    # TODO: check that models are actually the same!
    p_locs = p.particle_locations if isinstance(p, qinfer.ParticleDistribution) else p
    q_locs = q.particle_locations if isinstance(q, qinfer.ParticleDistribution) else q
    Q = p.model.Q if isinstance(p, qinfer.smc.SMCUpdater) else 1

    # Because the modelparam axis is last in each of the three cases, we're
    # good as far as broadcasting goes.
    delta = np.sqrt(Q) * (
        p_locs[:, np.newaxis, :] -
        q_locs[np.newaxis, :, :]
    )

    return np.sqrt(np.sum(delta**2, axis=-1))

def weighted_pairwise_distances(X, w, metric='euclidean', w_pow=0.5):
    r"""
    Given a feature matrix ``X`` with weights ``w``, calculates the modified
    distance metric :math:`\tilde{d}(p, q) = d(p, q) / (w(p) w(q) N^2)^p`, where
    :math:`N` is the length of ``X``. This metric is such that "heavy" feature
    vectors are considered to be closer to each other than "light" feature
    vectors, and are hence correspondingly less likely to be considered part of
    the same cluster.
    """

    if sklearn is None:
        raise ImportError("This function requires scikit-learn.")

    base_metric = sklearn.metrics.pairwise.pairwise_distances(X, metric=metric)
    N = w.shape[0]
    w_matrix = outer_product(w) * N**2

    return base_metric / (w_matrix ** w_pow)

## FINAL IMPORTS ##############################################################

import qinfer.smc
