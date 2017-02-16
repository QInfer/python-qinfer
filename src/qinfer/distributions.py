#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# distributions.py: module for probability distributions
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

## IMPORTS ###################################################################

from __future__ import division
from __future__ import absolute_import

from builtins import range
from future.utils import with_metaclass

import numpy as np
import scipy.stats as st
import scipy.linalg as la
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.spatial import ConvexHull, Delaunay

from functools import partial

import abc

from qinfer import utils as u
from qinfer.metrics import rescaled_distance_mtx
from qinfer.clustering import particle_clusters

import warnings

## EXPORTS ###################################################################

__all__ = [
    'Distribution',
    'SingleSampleMixin',
    'MixtureDistribution',
    'ParticleDistribution',
    'ProductDistribution',
    'UniformDistribution',
    'DiscreteUniformDistribution',
    'MVUniformDistribution',
    'ConstantDistribution',
    'NormalDistribution',
    'MultivariateNormalDistribution',
    'SlantedNormalDistribution',
    'LogNormalDistribution',
    'BetaDistribution',
    'BetaBinomialDistribution',
    'GammaDistribution',
    'GinibreUniform',
    'HaarUniform',
    'HilbertSchmidtUniform',
    'PostselectedDistribution',
    'ConstrainedSumDistribution',
    'InterpolatedUnivariateDistribution'
]

## FUNCTIONS #################################################################

def scipy_dist(name, *args, **kwargs):
    """
    Wraps calling a scipy.stats distribution to allow for pickling.
    See https://github.com/scipy/scipy/issues/3125.
    """
    return getattr(st, name)(*args, **kwargs)

## ABSTRACT CLASSES AND MIXINS ###############################################

class Distribution(with_metaclass(abc.ABCMeta, object)):
    """
    Abstract base class for probability distributions on one or more random
    variables.
    """

    @abc.abstractproperty
    def n_rvs(self):
        """
        The number of random variables that this distribution is over.

        :type: `int`
        """
        pass

    @abc.abstractmethod
    def sample(self, n=1):
        """
        Returns one or more samples from this probability distribution.

        :param int n: Number of samples to return.
        :rtype: numpy.ndarray
        :return: An array containing samples from the
            distribution of shape ``(n, d)``, where ``d`` is the number of
            random variables.
        """
        pass

class SingleSampleMixin(with_metaclass(abc.ABCMeta, object)):
    """
    Mixin class that extends a class so as to generate multiple samples
    correctly, given a method ``_sample`` that generates one sample at a time.
    """

    @abc.abstractmethod
    def _sample(self):
        pass

    def sample(self, n=1):
        samples = np.zeros((n, self.n_rvs))
        for idx in range(n):
            samples[idx, :] = self._sample()
        return samples


## CLASSES ###################################################################

class MixtureDistribution(Distribution):
    r"""
    Samples from a weighted list of distributions.

    :param weights: Length ``n_dist`` list or ``np.ndarray``
        of probabilites summing to 1.
    :param dist: Either a length ``n_dist`` list of ``Distribution`` instances,
        or a ``Distribution`` class, for example, ``NormalDistribution``.
        It is assumed that a list of ``Distribution``s all
        have the same ``n_rvs``.
    :param dist_args: If ``dist`` is a class, an array
        of shape ``(n_dist, n_rvs)`` where ``dist_args[k,:]`` defines
        the arguments of the k'th distribution. Use ``None`` if the distribution
        has no arguments.
    :param dist_kw_args: If ``dist`` is a class, a dictionary
        where each key's value is an array
        of shape ``(n_dist, n_rvs)`` where ``dist_kw_args[key][k,:]`` defines
        the keyword argument corresponding to ``key`` of the k'th distribution.
        Use ``None`` if the distribution needs no keyword arguments.
    :param bool shuffle: Whether or not to shuffle result after sampling. Not shuffling
        will result in variates being in the same order as
        the distributions. Default is ``True``.
    """

    def __init__(self, weights, dist, dist_args=None, dist_kw_args=None, shuffle=True):
        super(MixtureDistribution, self).__init__()
        self._weights = weights
        self._n_dist = len(weights)
        self._shuffle = shuffle

        try:
            self._example_dist = dist[0]
            self._is_dist_list = True
            self._dist_list = dist
            assert(self._n_dist == len(self._dist_list))
        except:
            self._is_dist_list = False
            self._dist = dist
            self._dist_args = dist_args
            self._dist_kw_args = dist_kw_args
            assert(self._n_dist == self._dist_args.shape[0])

            self._example_dist = self._dist(
                *self._dist_arg(0),
                **self._dist_kw_arg(0)
            )

    def _dist_arg(self, k):
        """
        Returns the arguments for the k'th distribution.

        :param int k: Index of distribution in question.
        :rtype: ``np.ndarary``
        """
        if self._dist_args is not None:
            return self._dist_args[k,:]
        else:
            return []

    def _dist_kw_arg(self, k):
        """
        Returns a dictionary of keyword arguments
        for the k'th distribution.

        :param int k: Index of the distribution in question.
        :rtype: ``dict``
        """
        if self._dist_kw_args is not None:
            return {
                key:self._dist_kw_args[key][k,:]
                for key in self._dist_kw_args.keys()
            }
        else:
            return {}

    @property
    def n_rvs(self):
        return self._example_dist.n_rvs

    @property
    def n_dist(self):
        """
        The number of distributions in the mixture distribution.
        """
        return self._n_dist

    def sample(self, n=1):
        # how many samples to take from each dist
        ns = np.random.multinomial(n, self._weights)
        idxs = np.arange(self.n_dist)[ns > 0]

        if self._is_dist_list:
            # sample from each distribution
            samples = np.concatenate([
                self._dist_list[k].sample(n=ns[k])
                for k in idxs
            ])
        else:
            # instantiate each distribution and then sample
            samples = np.concatenate([
                self._dist(
                        *self._dist_arg(k),
                        **self._dist_kw_arg(k)
                    ).sample(n=ns[k])
                for k in idxs
            ])

        # in-place shuffling
        if self._shuffle:
            np.random.shuffle(samples)

        return samples

class ParticleDistribution(Distribution):
    r"""
    A distribution consisting of a list of weighted vectors.
    Note that either `n_mps` or both (`particle_locations`, `particle_weights`)
    must be specified, or an error will be raised.

    :param numpy.ndarray particle_weights: Length ``n_particles`` list
        of particle weights.
    :param particle_locations: Shape ``(n_particles, n_mps)`` array of
        particle locations.
    :param int n_mps: Dimension of parameter space. This parameter should
        only be set when `particle_weights` and `particle_locations` are
        not set (and vice versa).
    """

    def __init__(self, n_mps=None, particle_locations=None, particle_weights=None):
        super(ParticleDistribution, self).__init__()
        if particle_locations is None or particle_weights is None:
            # Initialize with single particle at origin.
            self.particle_locations = np.zeros((1, n_mps))
            self.particle_weights = np.ones((1,))
        elif n_mps is None:
            self.particle_locations = particle_locations
            self.particle_weights = np.abs(particle_weights)
            self.particle_weights = self.particle_weights / np.sum(self.particle_weights)
        else:
            raise ValueError('Either the dimension of parameter space, `n_mps`, or the particles, `particle_locations` and `particle_weights` must be specified.')

    @property
    def n_particles(self):
        """
        Returns the number of particles in the distribution

        :type: `int`
        """
        return self.particle_locations.shape[0]

    @property
    def n_ess(self):
        """
        Returns the effective sample size (ESS) of the current particle
        distribution.

        :type: `float`
        :return: The effective sample size, given by :math:`1/\sum_i w_i^2`.
        """
        return 1 / (np.sum(self.particle_weights**2))

    ## DISTRIBUTION CONTRACT ##

    @property
    def n_rvs(self):
        """
        Returns the dimension of each particle.

        :type: `int`
        """
        return self.particle_locations.shape[1]

    def sample(self, n=1):
        """
        Returns random samples from the current particle distribution according
        to particle weights.

        :param int n: The number of samples to draw.
        :return: The sampled model parameter vectors.
        :rtype: `~numpy.ndarray` of shape ``(n, updater.n_rvs)``.
        """
        cumsum_weights = np.cumsum(self.particle_weights)
        return self.particle_locations[np.minimum(cumsum_weights.searchsorted(
            np.random.random((n,)),
            side='right'
        ), len(cumsum_weights) - 1)]

    ## MOMENT FUNCTIONS ##

    def est_mean(self):
        """
        Returns the mean value of the current particle distribution.

        :rtype: :class:`numpy.ndarray`, shape ``(n_mps,)``.
        :returns: An array containing the an estimate of the mean model vector.
        """
        return np.sum(
            # We need the particle index to be the rightmost index, so that
            # the two arrays align on the particle index as opposed to the
            # modelparam index.
            self.particle_weights * self.particle_locations.transpose([1, 0]),
            axis=1
        )

    def est_meanfn(self, fn):
        """
        Returns an the expectation value of a given function
        :math:`f` over the current particle distribution.

        Here, :math:`f` is represented by a function ``fn`` that is vectorized
        over particles, such that ``f(modelparams)`` has shape
        ``(n_particles, k)``, where ``n_particles = modelparams.shape[0]``, and
        where ``k`` is a positive integer.

        :param callable fn: Function implementing :math:`f` in a vectorized
            manner. (See above.)

        :rtype: :class:`numpy.ndarray`, shape ``(k, )``.
        :returns: An array containing the an estimate of the mean of :math:`f`.
        """

        return np.einsum('i...,i...',
            self.particle_weights, fn(self.particle_locations)
        )

    def est_covariance_mtx(self, corr=False):
        """
        Returns the full-rank covariance matrix of the current particle
        distribution.

        :param bool corr: If `True`, the covariance matrix is normalized
            by the outer product of the square root diagonal of the covariance matrix,
            i.e. the correlation matrix is returned instead.

        :rtype: :class:`numpy.ndarray`, shape
            ``(n_modelparams, n_modelparams)``.
        :returns: An array containing the estimated covariance matrix.
        """

        cov = u.particle_covariance_mtx(
            self.particle_weights,
            self.particle_locations)

        if corr:
            dstd = np.sqrt(np.diag(cov))
            cov /= (np.outer(dstd, dstd))

        return cov

    ## INFORMATION QUANTITIES ##

    def est_entropy(self):
        r"""
        Estimates the entropy of the current particle distribution
        as :math:`-\sum_i w_i \log w_i` where :math:`\{w_i\}`
        is the set of particles with nonzero weight.
        """
        nz_weights = self.particle_weights[self.particle_weights > 0]
        return -np.sum(np.log(nz_weights) * nz_weights)

    def _kl_divergence(self, other_locs, other_weights, kernel=None, delta=1e-2):
        """
        Finds the KL divergence between this and another particle
        distribution by using a kernel density estimator to smooth over the
        other distribution's particles.
        """
        if kernel is None:
            kernel = st.norm(loc=0, scale=1).pdf

        dist = rescaled_distance_mtx(self, other_locs) / delta
        K = kernel(dist)

        return -self.est_entropy() - (1 / delta) * np.sum(
            self.particle_weights *
            np.log(
                np.sum(
                    other_weights * K,
                    axis=1 # Sum over the particles of ``other``.
                )
            ),
            axis=0  # Sum over the particles of ``self``.
        )

    def est_kl_divergence(self, other, kernel=None, delta=1e-2):
        """
        Finds the KL divergence between this and another particle
        distribution by using a kernel density estimator to smooth over the
        other distribution's particles.

        :param SMCUpdater other:
        """
        return self._kl_divergence(
            other.particle_locations,
            other.particle_weights,
            kernel, delta
        )

    ## CLUSTER ESTIMATION METHODS #############################################

    def est_cluster_moments(self, cluster_opts=None):
        # TODO: document

        if cluster_opts is None:
            cluster_opts = {}

        for cluster_label, cluster_particles in particle_clusters(
                self.particle_locations, self.particle_weights,
                **cluster_opts
            ):

            w = self.particle_weights[cluster_particles]
            l = self.particle_locations[cluster_particles]
            yield (
                cluster_label,
                sum(w), # The zeroth moment is very useful here!
                u.particle_meanfn(w, l, lambda x: x),
                u.particle_covariance_mtx(w, l)
            )

    def est_cluster_covs(self, cluster_opts=None):
        # TODO: document

        cluster_moments = np.array(
            list(self.est_cluster_moments(cluster_opts=cluster_opts)),
            dtype=[
                ('label', 'int'),
                ('weight', 'float64'),
                ('mean', '{}float64'.format(self.n_rvs)),
                ('cov', '{0},{0}float64'.format(self.n_rvs)),
            ])

        ws = cluster_moments['weight'][:, np.newaxis, np.newaxis]

        within_cluster_var = np.sum(ws * cluster_moments['cov'], axis=0)
        between_cluster_var = u.particle_covariance_mtx(
            # Treat the cluster means as a new very small particle cloud.
            cluster_moments['weight'], cluster_moments['mean']
        )
        total_var = within_cluster_var + between_cluster_var

        return within_cluster_var, between_cluster_var, total_var

    def est_cluster_metric(self, cluster_opts=None):
        """
        Returns an estimate of how much of the variance in the current posterior
        can be explained by a separation between *clusters*.
        """
        wcv, bcv, tv = self.est_cluster_covs(cluster_opts)
        return np.diag(bcv) / np.diag(tv)

    ## REGION ESTIMATION METHODS ##############################################

    def est_credible_region(self, level=0.95, return_outside=False, modelparam_slice=None):
        """
        Returns an array containing particles inside a credible region of a
        given level, such that the described region has probability mass
        no less than the desired level.

        Particles in the returned region are selected by including the highest-
        weight particles first until the desired credibility level is reached.

        :param float level: Crediblity level to report.
        :param bool return_outside: If `True`, the return value is a tuple
            of the those particles within the credible region, and the rest
            of the posterior particle cloud.
        :param slice modelparam_slice: Slice over which model parameters
            to consider.

        :rtype: :class:`numpy.ndarray`, shape ``(n_credible, n_mps)``,
            where ``n_credible`` is the number of particles in the credible
            region and ``n_mps`` corresponds to the size of ``modelparam_slice``.
             If ``return_outside`` is ``True``, this method instead
             returns tuple ``(inside, outside)`` where ``inside`` is as
             described above, and ``outside`` has shape ``(n_particles-n_credible, n_mps)``.
        :return: An array of particles inside the estimated credible region. Or,
            if ``return_outside`` is ``True``, both the particles inside and the
            particles outside, as a tuple.
        """

        # which slice of modelparams to take
        s_ = np.s_[modelparam_slice] if modelparam_slice is not None else np.s_[:]
        mps = self.particle_locations[:, s_]

        # Start by sorting the particles by weight.
        # We do so by obtaining an array of indices `id_sort` such that
        # `particle_weights[id_sort]` is in descending order.
        id_sort = np.argsort(self.particle_weights)[::-1]

        # Find the cummulative sum of the sorted weights.
        cumsum_weights = np.cumsum(self.particle_weights[id_sort])

        # Find all the indices where the sum is less than level.
        # We first find id_cred such that
        # `all(cumsum_weights[id_cred] <= level)`.
        id_cred = cumsum_weights <= level
        # By construction, by adding the next particle to id_cred, it must be
        # true that `cumsum_weights[id_cred] >= level`, as required.
        id_cred[np.sum(id_cred)] = True

        # We now return a slice onto the particle_locations by first permuting
        # the particles according to the sort order, then by selecting the
        # credible particles.
        if return_outside:
            return (
                mps[id_sort][id_cred],
                mps[id_sort][np.logical_not(id_cred)]
            )
        else:
            return mps[id_sort][id_cred]

    def region_est_hull(self, level=0.95, modelparam_slice=None):
        """
        Estimates a credible region over models by taking the convex hull of
        a credible subset of particles.

        :param float level: The desired crediblity level (see
            :meth:`SMCUpdater.est_credible_region`).
        :param slice modelparam_slice: Slice over which model parameters
            to consider.

        :return: The tuple ``(faces, vertices)`` where ``faces`` describes all the
            vertices of all of the faces on the exterior of the convex hull, and
            ``vertices`` is a list of all vertices on the exterior of the
            convex hull.
        :rtype: ``faces`` is a ``numpy.ndarray`` with shape
            ``(n_face, n_mps, n_mps)`` and indeces ``(idx_face, idx_vertex, idx_mps)``
            where ``n_mps`` corresponds to the size of ``modelparam_slice``.
            ``vertices`` is an  ``numpy.ndarray`` of shape ``(n_vertices, n_mps)``.
        """
        points = self.est_credible_region(
            level=level,
            modelparam_slice=modelparam_slice
        )
        hull = ConvexHull(points)

        return points[hull.simplices], points[u.uniquify(hull.vertices.flatten())]

    def region_est_ellipsoid(self, level=0.95, tol=0.0001, modelparam_slice=None):
        r"""
        Estimates a credible region over models by finding the minimum volume
        enclosing ellipse (MVEE) of a credible subset of particles.

        :param float level: The desired crediblity level (see
            :meth:`SMCUpdater.est_credible_region`).
        :param float tol: The allowed error tolerance in the MVEE optimization
            (see :meth:`~qinfer.utils.mvee`).
        :param slice modelparam_slice: Slice over which model parameters
            to consider.

        :return: A tuple ``(A, c)`` where ``A`` is the covariance
            matrix of the ellipsoid and ``c`` is the center.
            A point :math:`\vec{x}` is in the ellipsoid whenever
            :math:`(\vec{x}-\vec{c})^{T}A^{-1}(\vec{x}-\vec{c})\leq 1`.
        :rtype: ``A`` is ``np.ndarray`` of shape ``(n_mps,n_mps)`` and
            ``centroid`` is ``np.ndarray`` of shape ``(n_mps)``.
            ``n_mps`` corresponds to the size of ``param_slice``.
        """
        _, vertices = self.region_est_hull(level=level, modelparam_slice=modelparam_slice)

        A, centroid = u.mvee(vertices, tol)
        return A, centroid

    def in_credible_region(self, points, level=0.95, modelparam_slice=None, method='hpd-hull', tol=0.0001):
        """
        Decides whether each of the points lie within a credible region
        of the current distribution.

        If ``tol`` is ``None``, the particles are tested directly against
        the convex hull object. If ``tol`` is a positive ``float``,
        particles are tested to be in the interior of the smallest
        enclosing ellipsoid of this convex hull, see
        :meth:`SMCUpdater.region_est_ellipsoid`.

        :param np.ndarray points: An ``np.ndarray`` of shape ``(n_mps)`` for
            a single point, or of shape ``(n_points, n_mps)`` for multiple points,
            where ``n_mps`` corresponds to the same dimensionality as ``param_slice``.
        :param float level: The desired crediblity level (see
            :meth:`SMCUpdater.est_credible_region`).
        :param str method: A string specifying which credible region estimator to
            use. One of ``'pce'``, ``'hpd-hull'`` or ``'hpd-mvee'`` (see below).
        :param float tol: The allowed error tolerance for those methods
            which require a tolerance (see :meth:`~qinfer.utils.mvee`).
        :param slice modelparam_slice: A slice describing which model parameters
            to consider in the credible region, effectively marginizing out the
            remaining parameters. By default, all model parameters are included.

        :return: A boolean array of shape ``(n_points, )`` specifying whether
            each of the points lies inside the confidence region.

        Methods
        ~~~~~~~

        The following values are valid for the ``method`` argument.

        - ``'pce'``: Posterior Covariance Ellipsoid.
            Computes the covariance
            matrix of the particle distribution marginalized over the excluded
            slices and uses the :math:`\chi^2` distribution to determine
            how to rescale it such the the corresponding ellipsoid has
            the correct size. The ellipsoid is translated by the
            mean of the particle distribution. It is determined which
            of the ``points`` are on the interior.
        - ``'hpd-hull'``: High Posterior Density Convex Hull.
            See :meth:`SMCUpdater.region_est_hull`. Computes the
            HPD region resulting from the particle approximation, computes
            the convex hull of this, and it is determined which
            of the ``points`` are on the interior.
        - ``'hpd-mvee'``: High Posterior Density Minimum Volume Enclosing Ellipsoid.
            See :meth:`SMCUpdater.region_est_ellipsoid`
            and :meth:`~qinfer.utils.mvee`. Computes the
            HPD region resulting from the particle approximation, computes
            the convex hull of this, and determines the minimum enclosing
            ellipsoid. Deterimines which
            of the ``points`` are on the interior.
        """

        if method == 'pce':
            s_ = np.s_[modelparam_slice] if modelparam_slice is not None else np.s_[:]
            A = self.est_covariance_mtx()[s_, s_]
            c = self.est_mean()[s_]
            # chi-squared distribution gives correct level curve conversion
            mult = st.chi2.ppf(level, c.size)
            results = u.in_ellipsoid(points, mult * A, c)

        elif method == 'hpd-mvee':
            tol = 0.0001 if tol is None else tol
            A, c = self.region_est_ellipsoid(level=level, tol=tol, modelparam_slice=modelparam_slice)
            results = u.in_ellipsoid(points, np.linalg.inv(A), c)

        elif method == 'hpd-hull':
            # it would be more natural to call region_est_hull,
            # but that function uses ConvexHull which has no
            # easy way of determining if a point is interior.
            # Here, Delaunay gives us access to all of the
            # necessary simplices.

            # this fills the convex hull with (n_mps+1)-dimensional
            # simplices; the convex hull is an almost-everywhere
            # disjoint union of these simplices
            hull = Delaunay(self.est_credible_region(level=level, modelparam_slice=modelparam_slice))

            # now we just check whether each of the given points are in
            # any of the simplices. (http://stackoverflow.com/a/16898636/1082565)
            results = hull.find_simplex(points) >= 0

        return results

class ProductDistribution(Distribution):
    r"""
    Takes a non-zero number of QInfer distributions :math:`D_k` as input
    and returns their Cartesian product.

    In other words, the returned distribution is
    :math:`\Pr(D_1, \dots, D_N) = \prod_k \Pr(D_k)`.

    :param Distribution factors:
        Distribution objects representing :math:`D_k`.
        Alternatively, one iterable argument can be given,
        in which case the factors are the values drawn from that iterator.
    """

    def __init__(self, *factors):
        if len(factors) == 1:
            try:
                self._factors = list(factors[0])
            except:
                self._factors = factors
        else:
            self._factors = factors

    @property
    def n_rvs(self):
        return sum([f.n_rvs for f in self._factors])

    def sample(self, n=1):
        return np.hstack([f.sample(n) for f in self._factors])


_DEFAULT_RANGES = np.array([[0, 1]])
_DEFAULT_RANGES.flags.writeable = False # Prevent anyone from modifying the
                                        # default ranges.


## CLASSES ###################################################################


class UniformDistribution(Distribution):
    """
    Uniform distribution on a given rectangular region.

    :param numpy.ndarray ranges: Array of shape ``(n_rvs, 2)``, where ``n_rvs``
        is the number of random variables, specifying the upper and lower limits
        for each variable.
    """

    def __init__(self, ranges=_DEFAULT_RANGES):
        if not isinstance(ranges, np.ndarray):
            ranges = np.array(ranges)

        if len(ranges.shape) == 1:
            ranges = ranges[np.newaxis, ...]

        self._ranges = ranges
        self._n_rvs = ranges.shape[0]
        self._delta = ranges[:, 1] - ranges[:, 0]

    @property
    def n_rvs(self):
        return self._n_rvs

    def sample(self, n=1):
        shape = (n, self._n_rvs)# if n == 1 else (self._n_rvs, n)
        z = np.random.random(shape)
        return self._ranges[:, 0] + z * self._delta

    def grad_log_pdf(self, var):
        # THIS IS NOT TECHNICALLY LEGIT; BCRB doesn't technically work with a
        # prior that doesn't go to 0 at its end points.  But we do it anyway.
        if var.shape[0] == 1:
            return 12/(self._delta)**2
        else:
            return np.zeros(var.shape)

class ConstantDistribution(Distribution):
    """
    Represents a determinstic variable; useful for combining with other
    distributions, marginalizing, etc.

    :param values: Shape ``(n,)`` array or list of values :math:`X_0` such that
        :math:`\Pr(X) = \delta(X - X_0)`.
    """

    def __init__(self, values):
        self._values = np.array(values)[np.newaxis, :]

    @property
    def n_rvs(self):
        return self._values.shape[1]

    def sample(self, n=1):
        return np.repeat(self._values, n, axis=0)

class NormalDistribution(Distribution):
    """
    Normal or truncated normal distribution over a single random
    variable.

    :param float mean: Mean of the represented random variable.
    :param float var: Variance of the represented random variable.
    :param tuple trunc: Limits at which the PDF of this
        distribution should be truncated, or ``None`` if
        the distribution is to have infinite support.
    """
    def __init__(self, mean, var, trunc=None):
        self.mean = mean
        self.var = var

        if trunc is not None:
            low, high = trunc
            sigma = np.sqrt(var)
            a = (low - mean) / sigma
            b = (high - mean) / sigma
            self.dist = partial(scipy_dist, 'truncnorm', a, b, loc=mean, scale=np.sqrt(var))
        else:
            self.dist = partial(scipy_dist, 'norm', mean, np.sqrt(var))

    @property
    def n_rvs(self):
        return 1

    def sample(self, n=1):
        return self.dist().rvs(size=n)[:, np.newaxis]

    def grad_log_pdf(self, x):
        return -(x - self.mean) / self.var

class MultivariateNormalDistribution(Distribution):
    """
    Multivariate (vector-valued) normal distribution.

    :param np.ndarray mean: Array of shape ``(n_rvs, )``
        representing the mean of the distribution.
    :param np.ndarray cov: Array of shape ``(n_rvs, n_rvs)``
        representing the covariance matrix of the distribution.
    """

    def __init__(self, mean, cov):

        # Flatten the mean first, so we have a strong guarantee about its
        # shape.

        self.mean = np.array(mean).flatten()
        self.cov = cov
        self.invcov = la.inv(cov)

    @property
    def n_rvs(self):
        return self.mean.shape[0]
    def sample(self, n=1):
        return np.einsum("ij,nj->ni", la.sqrtm(self.cov), np.random.randn(n, self.n_rvs)) + self.mean

    def grad_log_pdf(self, x):
        return -np.dot(self.invcov, (x - self.mean).transpose()).transpose()


class SlantedNormalDistribution(Distribution):
    r"""
    Uniform distribution on a given rectangular region  with
    additive noise. Random variates from this distribution
    follow :math:`X+Y` where :math:`X` is drawn uniformly
    with respect to the rectangular region defined by ranges, and
    :math:`Y` is normally distributed about 0 with variance
    ``weight**2``.

    :param numpy.ndarray ranges: Array of shape ``(n_rvs, 2)``, where ``n_rvs``
        is the number of random variables, specifying the upper and lower limits
        for each variable.
    :param float weight: Number specifying the inverse variance
        of the additive noise term.
    """

    def __init__(self, ranges=_DEFAULT_RANGES, weight=0.01):
        if not isinstance(ranges, np.ndarray):
            ranges = np.array(ranges)

        if len(ranges.shape) == 1:
            ranges = ranges[np.newaxis, ...]

        self._ranges = ranges
        self._n_rvs = ranges.shape[0]
        self._delta = ranges[:, 1] - ranges[:, 0]
        self._weight = weight

    @property
    def n_rvs(self):
        return self._n_rvs

    def sample(self, n=1):
        shape = (n, self._n_rvs)# if n == 1 else (self._n_rvs, n)
        z = np.random.randn(n, self._n_rvs)
        return self._ranges[:, 0] + \
                self._weight*z + \
                np.random.rand(n, self._n_rvs)*self._delta[np.newaxis,:]

class LogNormalDistribution(Distribution):
    """
    Log-normal distribution.

    :param mu: Location parameter (numeric), set to 0 by default.
    :param sigma: Scale parameter (numeric), set to 1 by default.
                  Must be strictly greater than zero.
    """

    def __init__(self, mu=0, sigma=1):
        self.mu = mu # lognormal location parameter
        self.sigma = sigma # lognormal scale parameter

        self.dist = partial(scipy_dist, 'lognorm', 1, mu, sigma) # scipy distribution location = 0

    @property
    def n_rvs(self):
        return 1

    def sample(self, n=1):
        return self.dist().rvs(size=n)[:, np.newaxis]

class BetaDistribution(Distribution):
    r"""
    The beta distribution, whose pdf at :math:`x` is proportional to
    :math:`x^{\alpha-1}(1-x)^{\beta-1}`.
    Note that either ``alpha`` and ``beta``, or ``mean`` and ``var``, must be
    specified as inputs;
    either case uniquely determines the distribution.

    :param float alpha: The alpha shape parameter of the beta distribution.
    :param float beta: The beta shape parameter of the beta distribution.
    :param float mean: The desired mean value of the beta distribution.
    :param float var: The desired variance of the beta distribution.
    """
    def __init__(self, alpha=None, beta=None, mean=None, var=None):
        if alpha is not None and beta is not None:
            self.alpha = alpha
            self.beta = beta
            self.mean = alpha / (alpha + beta)
            self.var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
        elif mean is not None and var is not None:
            self.mean = mean
            self.var = var
            self.alpha = mean ** 2 * (1 - mean) / var - mean
            self.beta = (1 - mean) ** 2 * mean / var - (1 - mean)
        else:
            raise ValueError(
                "BetaDistribution requires either (alpha and beta) "
                "or (mean and var)."
            )

        self.dist = st.beta(a=self.alpha, b=self.beta)

    @property
    def n_rvs(self):
        return 1

    def sample(self, n=1):
        return self.dist.rvs(size=n)[:, np.newaxis]

class BetaBinomialDistribution(Distribution):
    r"""
    The beta-binomial distribution, whose pmf at the non-negative
    integer :math:`k` is equal to
    :math:`\binom{n}{k}\frac{B(k+\alpha,n-k+\beta)}{B(\alpha,\beta)}`
    with :math:`B(\cdot,\cdot)` the beta function.
    This is the compound distribution whose variates are binomial distributed
    with a bias chosen from a beta distribution.
    Note that either ``alpha`` and ``beta``, or ``mean`` and ``var``, must be
    specified as inputs;
    either case uniquely determines the distribution.

    :param int n: The :math:`n` parameter of the beta-binomial distribution.
    :param float alpha: The alpha shape parameter of the beta-binomial distribution.
    :param float beta: The beta shape parameter of the beta-binomial distribution.
    :param float mean: The desired mean value of the beta-binomial distribution.
    :param float var: The desired variance of the beta-binomial distribution.
    """
    def __init__(self, n, alpha=None, beta=None, mean=None, var=None):

        self.n = n
        if alpha is not None and beta is not None:
            self.alpha = alpha
            self.beta = beta
            self.mean = n * alpha / (alpha + beta)
            self.var = n * alpha * beta * (alpha + beta + n) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        elif mean is not None and var is not None:
            self.mean = mean
            self.var = var
            self.alpha = - mean * (var + mean **2 - n * mean) / (mean ** 2 + n * (var - mean))
            self.beta = (n - mean) * (var + mean ** 2 - n * mean) / ((n - mean) * mean - n * var)
        else:
            raise ValueError("BetaBinomialDistribution requires either (alpha and beta) or (mean and var).")

        # Beta-binomial is a compound distribution, drawing binomial
        # RVs off of a beta-distrubuted bias.
        self._p_dist = st.beta(a=self.alpha, b=self.beta)

    @property
    def n_rvs(self):
        return 1

    def sample(self, n=1):
        p_vals = self._p_dist.rvs(size=n)[:, np.newaxis]
        # numpy.random.binomial supports sampling using different p values,
        # whereas scipy does not.
        return np.random.binomial(self.n, p_vals)

class GammaDistribution(Distribution):
    r"""
    The gamma distribution, whose pdf at :math:`x` is proportional to
    :math:`x^{-\alpha-1}e^{-x\beta}`.
    Note that either alpha and beta, or mean and var, must be
    specified as inputs;
    either case uniquely determines the distribution.

    :param float alpha: The alpha shape parameter of the gamma distribution.
    :param float beta: The beta shape parameter of the gamma distribution.
    :param float mean: The desired mean value of the gamma distribution.
    :param float var: The desired variance of the gamma distribution.
    """
    def __init__(self, alpha=None, beta=None, mean=None, var=None):
        if alpha is not None and beta is not None:
            self.alpha = alpha
            self.beta = beta
            self.mean = alpha / beta
            self.var = alpha / beta ** 2
        elif mean is not None and var is not None:
            self.mean = mean
            self.var = var
            self.alpha = mean ** 2 / var
            self.beta = mean / var
        else:
            raise ValueError("GammaDistribution requires either (alpha and beta) or (mean and var).")

        # This is the distribution we want up to a scale factor of beta
        self._dist = st.gamma(self.alpha)

    @property
    def n_rvs(self):
        return 1

    def sample(self, n=1):
        return self._dist.rvs(size=n)[:, np.newaxis] / self.beta

class MVUniformDistribution(Distribution):
    r"""
    Uniform distribution over the rectangle
    :math:`[0,1]^{\text{dim}}` with the restriction
    that vector must sum to 1. Equivalently, a
    uniform distribution over the ``dim-1`` simplex
    whose vertices are the canonical unit vectors of
    :math:`\mathbb{R}^\text{dim}`.

    :param int dim: Number of dimensions; ``n_rvs``.
    """

    def __init__(self, dim = 6):
        warnings.warn(
            "This class has been deprecated, and may "
            "be renamed in future versions.",
            DeprecationWarning
        )
        self._dim = dim

    @property
    def n_rvs(self):
        return self._dim

    def sample(self, n = 1):
        return np.random.mtrand.dirichlet(np.ones(self._dim),n)

class DiscreteUniformDistribution(Distribution):
    """
    Discrete uniform distribution over the integers between
    ``0`` and ``2**num_bits-1`` inclusive.

    :param int num_bits: non-negative integer specifying
        how big to make the interval.
    """

    def __init__(self, num_bits):
        self._num_bits = num_bits

    @property
    def n_rvs(self):
        return 1

    def sample(self, n=1):
        z = np.random.randint(2**self._num_bits,size=n)
        return z

class HilbertSchmidtUniform(SingleSampleMixin, Distribution):
    """
    Creates a new Hilber-Schmidt uniform prior on state space of dimension ``dim``.
    See e.g. [Mez06]_ and [Mis12]_.

    :param int dim: Dimension of the state space.
    """
    def __init__(self, dim=2):
        warnings.warn(
            "This class has been deprecated; please see "
            "qinfer.tomography.GinibreDistribution(rank=None).",
            DeprecationWarning
        )
        self.dim = dim
        self.paulis1Q = np.array([[[1,0],[0,1]],[[1,0],[0,-1]],[[0,-1j],[1j,0]],[[0,1],[1,0]]])

        self.paulis = self.make_Paulis(self.paulis1Q, 4)

    @property
    def n_rvs(self):
        return self.dim**2 - 1

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

        x = np.zeros([self.n_rvs])
        for idx in range(self.n_rvs):
            x[idx] = np.real(np.trace(np.dot(rho,self.paulis[idx+1])))

        return x

    def make_Paulis(self,paulis,d):
        if d == self.dim*2:
            return paulis
        else:
            temp = np.zeros([d**2,d,d],dtype='complex128')
            for idx in range(temp.shape[0]):
                temp[idx,:] = np.kron(paulis[np.trunc(idx/d)], self.paulis1Q[idx % 4])
            return self.make_Paulis(temp,d*2)


class HaarUniform(SingleSampleMixin, Distribution):
    """
    Haar uniform distribution of pure states of dimension ``dim``,
    parameterized as coefficients of the Pauli basis.

    :param int dim: Dimension of the state space.

    .. note::

        This distribution presently only works for ``dim==2`` and
        the Pauli basis.
    """
    def __init__(self, dim=2):
        warnings.warn(
            "This class has been deprecated; please see "
            "qinfer.tomography.GinibreDistribution(rank=1).",
            DeprecationWarning
        )
        # TODO: add basis as an option
        self.dim = dim


    @property
    def n_rvs(self):
        return 3

    def _sample(self):
        #Generate random unitary (see e.g. http://arxiv.org/abs/math-ph/0609050v2)
        z = (np.random.randn(self.dim,self.dim) + 1j*np.random.randn(self.dim,self.dim))/np.sqrt(2.0)
        q,r = la.qr(z)
        d = np.diag(r)

        ph = d/np.abs(d)
        ph = np.diag(ph)

        U = np.dot(q,ph)

        #TODO: generalize this to general dimensions
        #Apply Haar random unitary to |0> state to get random pure state
        psi = np.dot(U,np.array([1,0]))
        z = np.real(np.dot(psi.conj(),np.dot(np.array([[1,0],[0,-1]]),psi)))
        y = np.real(np.dot(psi.conj(),np.dot(np.array([[0,-1j],[1j,0]]),psi)))
        x = np.real(np.dot(psi.conj(),np.dot(np.array([[0,1],[1,0]]),psi)))

        return np.array([x,y,z])

class GinibreUniform(SingleSampleMixin, Distribution):
    """
    Creates a prior on state space of dimension dim according to the Ginibre
    ensemble with parameter ``k``.
    See e.g. [Mis12]_.

    :param int dim: Dimension of the state space.
    """
    def __init__(self,dim=2, k=2):
        warnings.warn(
            "This class has been deprecated; please see "
            "qinfer.tomography.GinibreDistribution.",
            DeprecationWarning
        )
        self.dim = dim
        self.k = k

    @property
    def n_rvs(self):
        return 3

    def _sample(self):
        #Generate random matrix
        z = np.random.randn(self.dim,self.k) + 1j*np.random.randn(self.dim,self.k)

        rho = np.dot(z,z.conj().transpose())
        rho = rho/np.trace(rho)

        z = np.real(np.trace(np.dot(rho,np.array([[1,0],[0,-1]]))))
        y = np.real(np.trace(np.dot(rho,np.array([[0,-1j],[1j,0]]))))
        x = np.real(np.trace(np.dot(rho,np.array([[0,1],[1,0]]))))

        return np.array([x,y,z])


class PostselectedDistribution(Distribution):
    """
    Postselects a distribution based on validity within a given model.
    """
    # TODO: rewrite LiuWestResampler in terms of this and a
    #       new MixtureDistribution.

    def __init__(self, distribution, model, maxiters=100):
        self._dist = distribution
        self._model = model

        self._maxiters = maxiters

    @property
    def n_rvs(self):
        return self._dist.n_rvs

    def sample(self, n=1):
        """
        Returns one or more samples from this probability distribution.

        :param int n: Number of samples to return.
        :return numpy.ndarray: An array containing samples from the
            distribution of shape ``(n, d)``, where ``d`` is the number of
            random variables.
        """
        samples = np.empty((n, self.n_rvs))
        idxs_to_sample = np.arange(n)

        iters = 0

        while idxs_to_sample.size and iters < self._maxiters:
            samples[idxs_to_sample] = self._dist.sample(len(idxs_to_sample))

            idxs_to_sample = idxs_to_sample[np.nonzero(np.logical_not(
                self._model.are_models_valid(samples[idxs_to_sample, :])
            ))[0]]

            iters += 1

        if idxs_to_sample.size:
            raise RuntimeError("Did not successfully postselect within {} iterations.".format(self._maxiters))

        return samples

    def grad_log_pdf(self, x):
        return self._dist.grad_log_pdf(x)

class InterpolatedUnivariateDistribution(Distribution):
    """
    Samples from a single-variable distribution specified by its PDF. The
    samples are drawn by first drawing uniform samples over the interval
    ``[0, 1]``, and then using an interpolation of the inverse-CDF
    corresponding to the given PDF to transform these samples into the
    desired distribution.

    :param callable pdf: Vectorized single-argument function that evaluates
        the PDF of the desired distribution.
    :param float compactification_scale: Scale of the compactified coordinates
        used to interpolate the given PDF.
    :param int n_interp_points: The number of points at which to sample the
        given PDF.
    """

    def __init__(self, pdf, compactification_scale=1, n_interp_points=1500):
        self._pdf = pdf
        self._xs  = u.compactspace(compactification_scale, n_interp_points)

        self._generate_interp()

    def _generate_interp(self):

        xs = self._xs

        pdfs = self._pdf(xs)
        norm_factor = np.trapz(pdfs, xs)

        self._cdfs = cumtrapz(pdfs / norm_factor, xs, initial=0)
        self._interp_inv_cdf = interp1d(self._cdfs, xs, bounds_error=False)

    @property
    def n_rvs(self):
        return 1

    def sample(self, n=1):
        return self._interp_inv_cdf(np.random.random(n))[:, np.newaxis]

class ConstrainedSumDistribution(Distribution):
    """
    Samples from an underlying distribution and then
    enforces that all samples must sum to some given
    value by normalizing each sample.

    :param Distribution underlying_distribution: Underlying probability distribution.
    :param float desired_total: Desired sum of each sample.
    """

    def __init__(self, underlying_distribution, desired_total=1):
        super(ConstrainedSumDistribution, self).__init__()
        self._ud = underlying_distribution
        self.desired_total = desired_total

    @property
    def underlying_distribution(self):
        return self._ud

    @property
    def n_rvs(self):
        return self.underlying_distribution.n_rvs

    def sample(self, n=1):
        s = self.underlying_distribution.sample(n)
        totals = np.sum(s, axis=1)[:,np.newaxis]
        return self.desired_total * np.sign(totals) * s / totals
