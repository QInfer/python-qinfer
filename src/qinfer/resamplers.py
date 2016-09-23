#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# resamplers.py: Implementations of various resampling algorithms.
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
    'Resampler',
    'LiuWestResampler'
]

## IMPORTS ####################################################################

import numpy as np
import scipy.linalg as la
import warnings

from .utils import outer_product, particle_meanfn, particle_covariance_mtx

from abc import ABCMeta, abstractmethod, abstractproperty
from future.utils import with_metaclass

import qinfer.clustering
from qinfer._exceptions import ResamplerWarning, ResamplerError

## LOGGING ####################################################################

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

## CLASSES ####################################################################

class Resampler(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def __call__(self,  model, particle_weights, particle_locations,
        n_particles=None,
        precomputed_mean=None, precomputed_cov=None
    ):
        """
        Resample the particles given by ``particle_weights`` and
        ``particle_locations``, drawing ``n_particles`` new particles.

        :param Model model: Model from which the particles are drawn,
            used to define the valid region for resampling.
        :param np.ndarray particle_weights: Weights of each particle,
            represented as an array of shape ``(n_original_particles, )``
            and dtype :obj:`float`.
        :param np.ndarray particle_locations: Locations of each particle,
            represented as an array of shape ``(n_original_particles,
            model.n_modelparams)`` and dtype :obj:`float`.
        :param int n_particles: Number of new particles to draw, or
            `None` to draw the same number as the original distribution.
        :param np.ndarray precomputed_mean: Mean of the original
            distribution, or `None` if this should be computed by the resampler.
        :param np.ndarray precomputed_cov: Covariance of the original
            distribution, or `None` if this should be computed by the resampler.

        :return np.ndarray new_weights: Weights of each new particle.
        :return np.ndarray new_locations: Locations of each new particle.        
        """

class ClusteringResampler(object):
    r"""
    Creates a resampler that breaks the particles into clusters, then applies
    a secondary resampling algorithm to each cluster independently.
    
    :param secondary_resampler: Resampling algorithm to be applied to each
        cluster. If ``None``, defaults to ``LiuWestResampler()``.
    """
    
    def __init__(self, eps=0.5, secondary_resampler=None, min_particles=5, metric='euclidean', weighted=False, w_pow=0.5, quiet=True):
        warnings.warn("This class is deprecated, and will be removed in a future version.", DeprecationWarning)
        self.secondary_resampler = (
            secondary_resampler
            if secondary_resampler is not None
            else LiuWestResampler()
        )
        
        self.eps = eps
        self.quiet = quiet
        self.min_particles = min_particles
        self.metric = metric
        self.weighted = weighted
        self.w_pow = w_pow
        
    ## METHODS ##
    
    def __call__(self, model, particle_weights, particle_locations):
        ## TODO: docstring.
        
        # Allocate new arrays to hold the weights and locations.        
        new_weights = np.empty(particle_weights.shape)
        new_locs    = np.empty(particle_locations.shape)
        
        # Loop over clusters, calling the secondary resampler for each.
        # The loop should include -1 if noise was found.
        for cluster_label, cluster_particles in clustering.particle_clusters(
                particle_locations, particle_weights,
                eps=self.eps, min_particles=self.min_particles, metric=self.metric,
                weighted=self.weighted, w_pow=self.w_pow,
                quiet=self.quiet
        ):
        
            # If we are resampling the NOISE label, we must use the global moments.
            if cluster_label == clustering.NOISE:
                extra_args = {
                    "precomputed_mean": particle_meanfn(particle_weights, particle_locations, lambda x: x),
                    "precomputed_cov":  particle_covariance_mtx(particle_weights, particle_locations)
                }
            else:
                extra_args = {}
            
            # Pass the particles in that cluster to the secondary resampler
            # and record the new weights and locations.
            cluster_ws, cluster_locs = self.secondary_resampler(model,
                particle_weights[cluster_particles],
                particle_locations[cluster_particles],
                **extra_args
            )
            
            # Renormalize the weights of each resampled particle by the total
            # weight of the cluster to which it belongs.
            cluster_ws /= np.sum(particle_weights[cluster_particles])
            
            # Store the updated cluster.
            new_weights[cluster_particles] = cluster_ws
            new_locs[cluster_particles]    = cluster_locs

        # Assert that we have not introduced any NaNs or Infs by resampling.
        assert np.all(np.logical_not(np.logical_or(
                np.isnan(new_locs), np.isinf(new_locs)
            )))
            
        return new_weights, new_locs

class LiuWestResampler(Resampler):
    r"""
    Creates a resampler instance that applies the algorithm of
    [LW01]_ to redistribute the particles.
    
    :param float a: Value of the parameter :math:`a` of the [LW01]_ algorithm
        to use in resampling.
    :param float h: Value of the parameter :math:`h` to use, or `None` to
        use that corresponding to :math:`a`.
    :param int maxiter: Maximum number of times to attempt to resample within
        the space of valid models before giving up.
    :param bool debug: Because the resampler can generate large amounts of
        debug information, nothing is output to the logger, even at DEBUG level,
        unless this flag is True.
    :param bool postselect: If `True`, ensures that models are valid by
        postselecting.
    :param float zero_cov_comp: Amount of covariance to be added to every
        parameter during resampling in the case that the estimated covariance
        has zero norm.
    :param callable kernel: Callable function ``kernel(*shape)`` that returns samples
        from a resampling distribution with mean 0 and variance 1.
        
    .. warning::
    
        The [LW01]_ algorithm preserves the first two moments of the
        distribution (in expectation over the random choices made by the
        resampler) if and only if :math:`a^2 + h^2 = 1`, as is set by the
        ``h=None`` keyword argument.
    """
    def __init__(self,
            a=0.98, h=None, maxiter=1000, debug=False, postselect=True,
            zero_cov_comp=1e-10,
            kernel=np.random.randn
        ):
        self.a = a # Implicitly calls the property setter below to set _h.
        if h is not None:
            self._override_h = True
            self._h = h
        self._maxiter = maxiter
        self._debug = debug
        self._postselect = postselect
        self._zero_cov_comp = zero_cov_comp
        self._kernel = kernel

    _override_h = False

    ## PROPERTIES ##

    @property
    def a(self):
        return self._a
        
    @a.setter
    def a(self, new_a):
        self._a = new_a
        if not self._override_h:
            self._h = np.sqrt(1 - new_a**2)

    ## METHODS ##
    
    def __call__(self, model, particle_weights, particle_locations,
        n_particles=None,
        precomputed_mean=None, precomputed_cov=None
    ):
        """
        Resample the particles according to algorithm given in 
        [LW01]_.
        """
        
        # Give shorter names to weights and locations.
        w, l = particle_weights, particle_locations
        
        # Possibly recompute moments, if not provided.
        if precomputed_mean is None:
            mean = particle_meanfn(w, l, lambda x: x)
        else:
            mean = precomputed_mean
        if precomputed_cov is None:
            cov = particle_covariance_mtx(w, l)
        else:
            cov = precomputed_cov
        
        if n_particles is None:
            n_particles = l.shape[0]
        
        # parameters in the Liu and West algorithm            
        a, h = self._a, self._h
        if la.norm(cov, 'fro') == 0:
            # The norm of the square root of S is literally zero, such that
            # the error estimated in the next step will not make sense.
            # We fix that by adding to the covariance a tiny bit of the
            # identity.
            warnings.warn(
                "Covariance has zero norm; adding in small covariance in "
                "resampler. Consider increasing n_particles to improve covariance "
                "estimates.",
                ResamplerWarning
            )
            cov = self._zero_cov_comp * np.eye(cov.shape[0])
        S, S_err = la.sqrtm(cov, disp=False)
        if not np.isfinite(S_err):
            raise ResamplerError(
                "Infinite error in computing the square root of the "
                "covariance matrix. Check that n_ess is not too small.")
        S = np.real(h * S)
        n_mp = l.shape[1]
        
        new_locs = np.empty((n_particles, n_mp))        
        cumsum_weights = np.cumsum(w)
        
        idxs_to_resample = np.arange(n_particles, dtype=int)
        
        # Preallocate js and mus so that we don't have rapid allocation and
        # deallocation.
        js = np.empty(idxs_to_resample.shape, dtype=int)
        mus = np.empty(new_locs.shape, dtype=l.dtype)
        
        # Loop as long as there are any particles left to resample.
        n_iters = 0
            
        # Draw j with probability self.particle_weights[j].
        # We do this by drawing random variates uniformly on the interval
        # [0, 1], then see where they belong in the CDF.
        js[:] = cumsum_weights.searchsorted(
            np.random.random((idxs_to_resample.size,)),
            side='right'
        )
        
        while idxs_to_resample.size and n_iters < self._maxiter:
            # Keep track of how many iterations we used.
            n_iters += 1
            
            # Set mu_i to a x_j + (1 - a) mu.
            mus[...] = a * l[js,:] + (1 - a) * mean
            
            # Draw x_i from N(mu_i, S).
            new_locs[idxs_to_resample, :] = mus + np.dot(S, self._kernel(n_mp, mus.shape[0])).T
            
            # Now we remove from the list any valid models.
            # We write it out in a longer form than is strictly necessary so
            # that we can validate assertions as we go. This is helpful for
            # catching models that may not hold to the expected postconditions.
            resample_locs = new_locs[idxs_to_resample, :]
            if self._postselect:
                valid_mask = model.are_models_valid(resample_locs)
            else:
                valid_mask = np.ones((resample_locs.shape[0],), dtype=bool)
            
            assert valid_mask.ndim == 1, "are_models_valid returned tensor, expected vector."
            
            n_invalid = np.sum(np.logical_not(valid_mask))
            
            if self._debug and n_invalid > 0:
                logger.debug(
                    "LW resampler found {} invalid particles; repeating.".format(
                        n_invalid
                    )
                )
            
            assert (
                (
                    len(valid_mask.shape) == 1
                    or len(valid_mask.shape) == 2 and valid_mask.shape[-1] == 1
                ) and valid_mask.shape[0] == resample_locs.shape[0]
            ), (
                "are_models_valid returned wrong shape {} "
                "for input of shape {}."
            ).format(valid_mask.shape, resample_locs.shape)
            
            idxs_to_resample = idxs_to_resample[np.nonzero(np.logical_not(
                valid_mask
            ))[0]]

            # This may look a little weird, but it should delete the unused
            # elements of js, so that we don't need to reallocate.
            js = js[np.logical_not(valid_mask)]
            mus = mus[:idxs_to_resample.size, :]
            
        if idxs_to_resample.size:
            # We failed to force all models to be valid within maxiter attempts.
            # This means that we could be propagating out invalid models, and
            # so we should warn about that.
            warnings.warn((
                "Liu-West resampling failed to find valid models for {} "
                "particles within {} iterations."
            ).format(idxs_to_resample.size, self._maxiter), ResamplerWarning)
            
        if self._debug:
            logger.debug("LW resampling completed in {} iterations.".format(n_iters))

        # Now we reset the weights to be uniform, letting the density of
        # particles represent the information that used to be stored in the
        # weights. This is done by SMCUpdater, and so we simply need to return
        # the new locations here.
        return np.ones((w.shape[0],)) / w.shape[0], new_locs
        
    
