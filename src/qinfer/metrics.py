#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# metrics.py: Metrics for use with SciPy and sklearn functions.
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
