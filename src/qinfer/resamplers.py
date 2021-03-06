#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# resamplers.py: Implementations of various resampling algorithms.
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
    'Resampler',
    'LiuWestResampler'
]

## IMPORTS ####################################################################

import numpy as np
import scipy.linalg as la
import warnings

from ._due import due, BibTeX
from .utils import outer_product, particle_meanfn, particle_covariance_mtx, sqrtm_psd

from abc import ABCMeta, abstractmethod, abstractproperty
from future.utils import with_metaclass

import qinfer.clustering
from qinfer._exceptions import ResamplerWarning, ResamplerError
from qinfer.distributions import ParticleDistribution

## LOGGING ####################################################################

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

## CLASSES ####################################################################

class Resampler(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def __call__(self,  model, particle_dist,
        n_particles=None,
        precomputed_mean=None, precomputed_cov=None
    ):
        """
        Resample the particles given by ``particle_weights`` and
        ``particle_locations``, drawing ``n_particles`` new particles.

        :param Model model: Model from which the particles are drawn,
            used to define the valid region for resampling.
        :param ParticleDistribution paricle_dist: The particle distribution to
            be resampled.
        :param int n_particles: Number of new particles to draw, or
            `None` to draw the same number as the original distribution.
        :param np.ndarray precomputed_mean: Mean of the original
            distribution, or `None` if this should be computed by the resampler.
        :param np.ndarray precomputed_cov: Covariance of the original
            distribution, or `None` if this should be computed by the resampler.

        :return ParticleDistribution: Resampled particle distribution
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
    :param int default_n_particles: The default number of particles to draw during
        a resampling action. If ``None``, the number of redrawn particles
        redrawn will be equal to the number of particles given.
        The value of ``default_n_particles`` can be overridden by any integer
        value of ``n_particles`` given to ``__call__``.


    .. warning::

        The [LW01]_ algorithm preserves the first two moments of the
        distribution (in expectation over the random choices made by the
        resampler) if and only if :math:`a^2 + h^2 = 1`, as is set by the
        ``h=None`` keyword argument.
    """

    @due.dcite(
        BibTeX("""
            @incollection{liu_combined_2001,
                title = {Combined Parameter and State Estimation in Simulation-Based Filtering},
                timestamp = {2013-01-28T21:57:35Z},
                urldate = {2013-01-28},
                booktitle = {Sequential {Monte Carlo} Methods in Practice},
                publisher = {{Springer-Verlag, New York}},
                author = {Liu, Jane and West, Mike},
                editor = {De Freitas and Gordon, NJ},
                year = {2001}
            }
        """),
        description="Liu-West resampler",
        tags=['implementation']
    )
    def __init__(self,
            a=0.98, h=None, maxiter=1000, debug=False, postselect=True,
            zero_cov_comp=1e-10,
            default_n_particles=None,
            kernel=np.random.randn
        ):
        self._default_n_particles = default_n_particles
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

    def __call__(self, model, particle_dist,
        n_particles=None,
        precomputed_mean=None, precomputed_cov=None
    ):
        """
        Resample the particles according to algorithm given in
        [LW01]_.
        """

        # Possibly recompute moments, if not provided.
        if precomputed_mean is None:
            mean = particle_dist.est_mean()
        else:
            mean = precomputed_mean
        if precomputed_cov is None:
            cov = particle_dist.est_covariance_mtx()
        else:
            cov = precomputed_cov

        if n_particles is None:
            if self._default_n_particles is None:
                n_particles = particle_dist.n_particles
            else:
                n_particles = self._default_n_particles

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
        S, S_err = sqrtm_psd(cov)
        if not np.isfinite(S_err):
            raise ResamplerError(
                "Infinite error in computing the square root of the "
                "covariance matrix. Check that n_ess is not too small.")
        S = np.real(h * S)

        # Give shorter names to weights, locations, and nr. of random variables
        w = particle_dist.particle_weights
        l = particle_dist.particle_locations
        n_rvs = particle_dist.n_rvs

        new_locs = np.empty((n_particles, n_rvs))
        cumsum_weights = np.cumsum(w)

        idxs_to_resample = np.arange(n_particles, dtype=int)

        # Loop as long as there are any particles left to resample.
        n_iters = 0

        # Draw j with probability self.particle_weights[j].
        # We do this by drawing random variates uniformly on the interval
        # [0, 1], then see where they belong in the CDF.
        js = cumsum_weights.searchsorted(
            np.random.random((idxs_to_resample.size,)),
            side='right'
        )

        # Set mu_i to a x_j + (1 - a) mu.
        # FIXME This should use particle_dist.particle_mean
        mus = a * l[js,:] + (1 - a) * mean

        while idxs_to_resample.size and n_iters < self._maxiter:
            # Keep track of how many iterations we used.
            n_iters += 1

            # Draw x_i from N(mu_i, S).
            new_locs[idxs_to_resample, :] = mus + np.dot(S, self._kernel(n_rvs, mus.shape[0])).T

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
        new_weights = np.ones((n_particles,)) / n_particles
        return ParticleDistribution(particle_locations=new_locs,
                                    particle_weights=new_weights)
