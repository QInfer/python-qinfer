#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# smc.py: Sequential Monte Carlo module
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

from __future__ import division

## ALL ########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
    'SMCUpdater',
    'SMCUpdaterBCRB',
    'SMCUpdaterABC'
]

## IMPORTS ####################################################################

import warnings

import numpy as np

from scipy.spatial import Delaunay
import scipy.linalg as la
import scipy.stats

from qinfer.abstract_model import DifferentiableModel
from qinfer.metrics import rescaled_distance_mtx
from qinfer import clustering
from qinfer.distributions import Distribution
from qinfer.resamplers import LiuWestResampler
from qinfer.utils import outer_product, mvee, uniquify, particle_meanfn, \
        particle_covariance_mtx, format_uncertainty
from qinfer._exceptions import ApproximationWarning

try:
    import matplotlib.pyplot as plt
except ImportError:
    import warnings
    warnings.warn("Could not import pyplot. Plotting methods will not work.")
    plt = None

## CLASSES #####################################################################

class SMCUpdater(Distribution):
    r"""
    Creates a new Sequential Monte carlo updater, using the algorithm of
    [GFWC12]_.

    :param qinfer.abstract_model.Model model: Model whose parameters are to be inferred.
    :param int n_particles: The number of particles to be used in the particle approximation.
    :param qinfer.distributions.Distribution prior: A representation of the prior distribution.
    :param callable resampler: Specifies the resampling algorithm to be used. See :ref:`resamplers`
        for more details.
    :param float resample_thresh: Specifies the threshold for :math:`N_{\text{ess}}` to decide when to resample.
    """
    def __init__(self,
            model, n_particles, prior,
            resample_a=None, resampler=None, resample_thresh=0.5
            ):

        self._resample_count = 0
        
        self.model = model
        self.n_particles = n_particles
        self.prior = prior

        ## RESAMPLER CONFIGURATION ##
        # Backward compatibility with the old resample_a keyword argument,
        # which assumed that the Liu and West resampler was being used.
        if resample_a is not None:
            warnings.warn("The 'resample_a' keyword argument is deprecated; use 'resampler=LiuWestResampler(a)' instead.", DeprecationWarning)
            if resampler is not None:
                raise ValueError("Both a resample_a and an explicit resampler were provided; please provide only one.")
            self.resampler = LiuWestResampler(a=resample_a)
        else:
            if resampler is None:
                self.resampler = LiuWestResampler()
            else:
                self.resampler = resampler


        self.resample_thresh = resample_thresh

        self._data_record = []
        self._normalization_record = []
        
        ## PARTICLE INITIALIZATION ##
        # Particles are stored using two arrays, particle_locations and
        # particle_weights, such that:
        # 
        # particle_locations[idx_particle, idx_modelparam] is the idx_modelparam
        #     parameter of the particle idx_particle.
        # particle_weights[idx_particle] is the weight of the particle
        #     idx_particle.
        self.particle_locations = np.zeros((n_particles, model.n_modelparams))
        self.particle_weights = np.ones((n_particles,)) / n_particles

        for idx_particle in xrange(n_particles):
            self.particle_locations[idx_particle, :] = prior.sample()

    ## PROPERTIES #############################################################

    @property
    def resample_count(self):
        """
        Returns the number of times that the updater has resampled the particle
        approximation.
        
        :rtype: `int`
        """
        # We wrap this in a property to prevent external resetting and to enable
        # a docstring.
        return self._resample_count

    @property
    def normalization_record(self):
        """
        Returns the normalization record.
        
        :rtype: `float`
        """
        # We wrap this in a property to prevent external resetting and to enable
        # a docstring.
        return self._normalization_record
        
    @property
    def log_total_likelihood(self):
        """
        Returns the log-likelihood of all the data collected so far.
        
        Equivalent to::
            
            np.sum(np.log(updater.normalization_record))
        
        :rtype: `float`
        """
        return np.sum(np.log(self.normalization_record))
        
    @property
    def n_ess(self):
        """
        Estimates the effective sample size (ESS) of the current distribution
        over model parameters.

        :return float: The effective sample size, given by :math:`1/\sum_i w_i^2`.
        """
        return 1 / (np.sum(self.particle_weights**2))

    @property
    def data_record(self):
        # TODO: return read-only view onto the data record.
        return self._data_record

    ## PRIVATE METHODS ########################################################
    
    def _maybe_resample(self):
        """
        Checks the resample threshold and conditionally resamples.
        """
        if self.n_ess < self.n_particles * self.resample_thresh:
            self.resample()
            pass

    ## UPDATE METHODS #########################################################

    def hypothetical_update(self, outcomes, expparams, return_likelihood=False, return_normalization=False):
        """
        Produces the particle weights for the posterior of a hypothetical
        experiment.

        :param outcomes: Integer index of the outcome of the hypothetical
            experiment.
            TODO: Fix this to take an array-like of ints as well.
        :type outcomes: int or an ndarray of dtype int.
        :param expparams: TODO

        :type weights: ndarray, shape (n_outcomes, n_expparams, n_particles)
        :param weights: Weights assigned to each particle in the posterior
            distribution :math:`\Pr(\omega | d)`.
        """

        # It's "hypothetical", don't want to overwrite old weights yet!
        weights = self.particle_weights
        locs = self.particle_locations

        # Check if we have a single outcome or an array. If we only have one
        # outcome, wrap it in a one-index array.
        if not isinstance(outcomes, np.ndarray):
            outcomes = np.array([outcomes])

        # update the weights sans normalization
        # Rearrange so that likelihoods have shape (outcomes, experiments, models).
        # This makes the multiplication with weights (shape (models,)) make sense,
        # since NumPy broadcasting rules align on the right-most index.
        L = self.model.likelihood(outcomes, locs, expparams).transpose([0, 2, 1])
        hyp_weights = weights * L
        
        # Sum up the weights to find the renormalization scale.
        norm_scale = np.sum(hyp_weights, axis=2)[..., np.newaxis]
        
        # As a special case, check whether any entries of the norm_scale
        # are zero. If this happens, that implies that all of the weights are
        # zero--- that is, that the hypothicized outcome was impossible.
        # Conditioned on an impossible outcome, all of the weights should be
        # zero. To allow this to happen without causing a NaN to propagate,
        # we forcibly set the norm_scale to 1, so that the weights will
        # all remain zero.
        #
        # We don't actually want to propagate this out to the caller, however,
        # and so we save the "fixed" norm_scale to a new array.
        fixed_norm_scale = norm_scale.copy()
        fixed_norm_scale[np.abs(norm_scale) < np.spacing(0)] = 1
        
        # normalize
        norm_weights = hyp_weights / fixed_norm_scale
            # Note that newaxis is needed to align the two matrices.
            # This introduces a length-1 axis for the particle number,
            # so that the normalization is broadcast over all particles.
        if not return_likelihood:
            if not return_normalization:
                return norm_weights
            else:
                return norm_weights, norm_scale
        else:
            if not return_normalization:
                return norm_weights, L
            else:
                return norm_weights, L, norm_scale

    def update(self, outcome, expparams, check_for_resample=True):
        """
        Given an experiment and an outcome of that experiment, updates the
        posterior distribution to reflect knowledge of that experiment.

        After updating, resamples the posterior distribution if necessary.

        :param int outcome: Label for the outcome that was observed, as defined
            by the :class:`~qinfer.abstract_model.Model` instance under study.
        :param expparams: Parameters describing the experiment that was
            performed.
        :type expparams: :class:`~numpy.ndarray` of dtype given by the
            :attr:`~qinfer.abstract_model.Model.expparams_dtype` property
            of the underlying model
        :param bool check_for_resample: If :obj:`True`, after performing the
            update, the effective sample size condition will be checked and
            a resampling step may be performed.
        """

        # First, record the outcome.
        # TODO: record the experiment as well.
        self._data_record.append(outcome)

        # Perform the update      
        weights, norm = self.hypothetical_update(outcome, expparams, return_normalization=True)

        # Since hypothetical_update returns an array indexed by
        # [outcome, experiment, particle], we need to strip off those two
        # indices first.
        self.particle_weights[:] = weights[0,0,:]
        
        # Record the normalization
        self._normalization_record.append(norm[0][0])

        if check_for_resample:
            self._maybe_resample()
            
        if not np.all(self.particle_weights >= 0):
            warnings.warn("Negative weights occured in particle approximation. Smallest weight observed == {}. Clipping weights.".format(np.min(self.particle_weights)), ApproximationWarning)
            np.clip(self.particle_weights, 0, 1, out=self.particle_weights)

    def batch_update(self, outcomes, expparams, resample_interval=5):
        r"""
        Updates based on a batch of outcomes and experiments, rather than just
        one.

        :param numpy.ndarray outcomes: An array of outcomes of the experiments that
            were performed.
        :param numpy.ndarray expparams: Either a scalar or record single-index
            array of experiments that were performed.
        :param int resample_interval: Controls how often to check whether
            :math:`N_{\text{ess}}` falls below the resample threshold.
        """

        # TODO: write a faster implementation here using vectorized calls to
        #       likelihood.

        # Check that the number of outcomes and experiments is the same.
        n_exps = outcomes.shape[0]
        if expparams.shape[0] != n_exps:
            raise ValueError("The number of outcomes and experiments must match.")

        # Loop over experiments and update one at a time.
        for idx_exp, (outcome, experiment) in enumerate(izip(iter(outcomes), iter(expparams))):
            self.update(outcome, experiment, check_for_resample=False)
            if (idx_exp + 1) % resample_interval == 0:
                self._maybe_resample()

    ## RESAMPLING METHODS #####################################################

    def resample(self):
        # TODO: add amended docstring.

        # Record that we have performed a resampling step.
        self._resample_count += 1

        # Find the new particle locations according to the chosen resampling
        # algorithm.
        # We pass the model so that the resampler can check for validity of
        # newly placed particles.
        self.particle_weights, self.particle_locations = \
            self.resampler(self.model, self.particle_weights, self.particle_locations)

        # Reset the weights to uniform.
        self.particle_weights[:] = (1/self.n_particles)


    ## DISTRIBUTION CONTRACT ##################################################
    
    @property
    def n_rvs(self):
        return self._model.n_modelparams
        
    def sample(self, n=1):
        # TODO: cache this.
        cumsum_weights = np.cumsum(self.particle_weights)
        return self.particle_locations[cumsum_weights.searchsorted(
            np.random.random((n,)),
            side='right'
        )]

    ## ESTIMATION METHODS #####################################################

    def est_mean(self):
        """
        Returns an estimate of the posterior mean model, given by the
        expectation value over the current SMC approximation of the posterior
        model distribution.
        
        :rtype: :class:`numpy.ndarray`, shape ``(n_modelparams,)``.
        :returns: An array containing the an estimate of the mean model vector.
        """
        return np.sum(
            # We need the particle index to be the rightmost index, so that
            # the two arrays align on the particle index as opposed to the
            # modelparam index.
            self.particle_weights * self.particle_locations.transpose([1, 0]),
            # The argument now has shape (n_modelparams, n_particles), so that
            # the sum should collapse the particle index, 1.
            axis=1
        )
        
    def est_meanfn(self, fn):
        """
        Returns an estimate of the expectation value of a given function
        :math:`f` of the model parameters, given by a sum over the current SMC
        approximation of the posterior distribution over models.
        
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

    def est_covariance_mtx(self):
        """
        Returns an estimate of the covariance of the current posterior model
        distribution, given by the covariance of the current SMC approximation.
        
        :rtype: :class:`numpy.ndarray`, shape
            ``(n_modelparams, n_modelparams)``.
        :returns: An array containing the estimated covariance matrix.
        """
        return particle_covariance_mtx(
            self.particle_weights,
            self.particle_locations)

    def bayes_risk(self, expparams):
        r"""
        Calculates the Bayes risk for a hypothetical experiment, assuming the
        quadratic loss function defined by the current model's scale matrix
        (see :attr:`qinfer.abstract_model.Simulatable.Q`).
        
        :param expparams: The experiment at which to compute the Bayes risk.
        :type expparams: :class:`~numpy.ndarray` of dtype given by the current
            model's :attr:`~qinfer.abstract_model.Simulatable.expparams_dtype` property,
            and of shape ``(1,)``
            
        :return float: The Bayes risk for the current posterior distribution
            of the hypothetical experiment ``expparams``.
        """
        # This subroutine computes the bayes risk for a hypothetical experiment
        # defined by expparams.

        # Assume expparams is a single experiment

        # expparams =
        # Q = np array(Nmodelparams), which contains the diagonal part of the
        #     rescaling matrix.  Non-diagonal could also be considered, but
        #     for the moment this is not implemented.
        nout = self.model.n_outcomes(expparams) # This is a vector so this won't work
        w, L = self.hypothetical_update(np.arange(nout), expparams, return_likelihood=True)
        w = w[:, 0, :] # Fix w.shape == (n_outcomes, n_particles).
        L = L[:, :, 0] # Fix L.shape == (n_outcomes, n_particles).

        xs = self.particle_locations.transpose([1, 0]) # shape (n_mp, n_particles).
        
        # In the following, we will use the subscript convention that
        # "o" refers to an outcome, "p" to a particle, and
        # "i" to a model parameter.
        # Thus, mu[o,i] is the sum over all particles of w[o,p] * x[i,p].
        mu = np.einsum('op,ip', w, xs)
        
        var = (
            # This sum is a reduction over the particle index and thus
            # represents an expectation value over the diagonal of the
            # outer product $x . x^T$.
            np.einsum('op,ip', w, xs**2)
            # We finish by subracting from the above expectation value
            # the diagonal of the outer product $mu . mu^T$.
            - mu**2).T


        rescale_var = np.sum(self.model.Q * var, axis=1)
        # Q has shape (n_mp,), therefore rescale_var has shape (n_outcomes,).
        tot_like = np.sum(L, axis=1)
        return np.dot(tot_like.T, rescale_var)
        
    def expected_information_gain(self, expparams):
        r"""
        Calculates the expected information gain for a hypothetical experiment.
        
        :param expparams: The experiment at which to compute expected
            information gain.
        :type expparams: :class:`~numpy.ndarray` of dtype given by the current
            model's :attr:`~qinfer.abstract_model.Simulatable.expparams_dtype` property,
            and of shape ``(1,)``
            
        :return float: The Bayes risk for the current posterior distribution
            of the hypothetical experiment ``expparams``.
        """

        nout = self.model.n_outcomes(expparams)
        w, L = self.hypothetical_update(np.arange(nout), expparams, return_likelihood=True)
        w = w[:, 0, :] # Fix w.shape == (n_outcomes, n_particles).
        L = L[:, :, 0] # Fix L.shape == (n_outcomes, n_particles).
        
        # This is a special case of the KL divergence estimator (see below),
        # in which the other distribution is guaranteed to share support.
        #
        # KLD[idx_outcome] = Sum over particles(self * log(self / other[idx_outcome])
        # Est. KLD = E[KLD[idx_outcome] | outcomes].
        
        KLD = np.sum(
            self.particle_weights * np.log(self.particle_weights / w),
            axis=1 # Sum over particles.
        )
        
        tot_like = np.sum(L, axis=1)
        return np.dot(tot_like, KLD)
        
    def est_entropy(self):
        nz_weights = self.particle_weights[self.particle_weights > 0]
        return -np.sum(np.log(nz_weights) * nz_weights)
        
    def est_kl_divergence(self, other, kernel=None, delta=1e-2):
        # TODO: document.
        if kernel is None:
            kernel = scipy.stats.norm(loc=0, scale=1).pdf
        
        dist = rescaled_distance_mtx(self, other) / delta
        K = kernel(dist)
        
        
        return -self.est_entropy() - (1 / delta) * np.sum(
            self.particle_weights * 
            np.log(
                np.sum(
                    other.particle_weights * K,
                    axis=1 # Sum over the particles of ``other``.
                )
            ),
            axis=0  # Sum over the particles of ``self``.
        )
        
    ## CLUSTER ESTIMATION METHODS #############################################
        
    def est_cluster_moments(self, cluster_opts=None):
        # TODO: document
        
        if cluster_opts is None:
            cluster_opts = {}
        
        for cluster_label, cluster_particles in clustering.particle_clusters(
                self.particle_locations, self.particle_weights,
                **cluster_opts
            ):
            
            w = self.particle_weights[cluster_particles]
            l = self.particle_locations[cluster_particles]
            yield (
                cluster_label,
                sum(w), # The zeroth moment is very useful here!
                particle_meanfn(w, l, lambda x: x),
                particle_covariance_mtx(w, l)
            )
            
    def est_cluster_covs(self, cluster_opts=None):
        # TODO: document
        
        cluster_moments = np.array(
            list(self.est_cluster_moments(cluster_opts)),
            dtype=[
                ('label', 'int'),
                ('weight', 'float64'),
                ('mean', '{}float64'.format(self.model.n_modelparams)),
                ('cov', '{0},{0}float64'.format(self.model.n_modelparams)),
            ])
            
        ws = cluster_moments['weight'][:, np.newaxis, np.newaxis]
            
        within_cluster_var = np.sum(ws * cluster_moments['cov'], axis=0)
        between_cluster_var = particle_covariance_mtx(
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

    def est_credible_region(self, level=0.95):
        """
        Returns an array containing particles inside a credible region of a
        given level, such that the described region has probability mass
        no less than the desired level.
        
        Particles in the returned region are selected by including the highest-
        weight particles first until the desired credibility level is reached.
        
        :rtype: :class:`numpy.ndarray`, shape ``(n_credible, n_modelparams)``,
            where ``n_credible`` is the number of particles in the credible
            region
        :returns: An array of particles inside the estimated credible region.
        """
        
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
        return self.particle_locations[id_sort][id_cred]
    
    def region_est_hull(self, level=0.95):
        """
        Estimates a credible region over models by taking the convex hull of
        a credible subset of particles.
        
        :param float level: The desired crediblity level (see
            :meth:`SMCUpdater.est_credible_region`).
        """
        # TODO: document return values.
        points = self.est_credible_region(level = level)
        tri = Delaunay(points)
        faces = []
        hull = tri.convex_hull
        
        for ia, ib, ic in hull:
            faces.append(points[[ia, ib, ic]])    

        vertices = points[uniquify(hull.flatten())]
        
        return faces, vertices

    def region_est_ellipsoid(self, level=0.95, tol=0.0001):
        """
        Estimates a credible region over models by finding the minimum volume
        enclosing ellipse (MVEE) of a credible subset of particles.
        
        
        :param float level: The desired crediblity level (see
            :meth:`SMCUpdater.est_credible_region`).
        :param float tol: The allowed error tolerance in the MVEE optimization
            (see :meth:`~qinfer.utils.mvee`).
        """
        # TODO: document return values.
        faces, vertices = self.region_est_hull(level=level)
                
        A, centroid = mvee(vertices, tol)
        return A, centroid
        
    ## MISC METHODS ###########################################################
    
    def risk(self, x0):
        return self.bayes_risk(np.array([(x0,)], dtype=self.model.expparams_dtype))
        
    ## PLOTTING METHODS #######################################################
    
    def posterior_mesh(self, idx_param1=0, idx_param2=1, res1=100, res2=100, smoothing=0.01):
        """
        Returns a mesh, useful for plotting, of kernel density estimation
        of a 2D projection of the current posterior distribution.
        
        :param int idx_param1: Parameter to be treated as :math:`x` when
            plotting.
        :param int idx_param2: Parameter to be treated as :math:`y` when
            plotting.
        :param int res1: Resolution along the :math:`x` direction.
        :param int res2: Resolution along the :math:`y` direction.
        :param float smoothing: Standard deviation of the Gaussian kernel
            used to smooth the particle approximation to the current posterior.
            
        .. seealso::
        
            :meth:`SMCUpdater.plot_posterior_contour`
        """
        # WARNING: fancy indexing is used here, which means that a copy is
        #          made.
        locs = self.particle_locations[:, [idx_param1, idx_param2]]
    
        p1s, p2s = np.meshgrid(
            np.linspace(np.min(locs[:, 0]), np.max(locs[:, 0]), res1),
            np.linspace(np.min(locs[:, 1]), np.max(locs[:, 1]), res2)
        )
        plot_locs = np.array([p1s, p2s]).T.reshape((np.prod(p1s.shape), 2))
        
        pr = np.sum( # <- sum over the particles in the SMC approximation.
            np.prod( # <- product over model parameters to get a multinormal
                # Evaluate the PDF at the plotting locations, with a normal
                # located at the particle locations.
                scipy.stats.norm.pdf(
                    plot_locs[:, np.newaxis, :],
                    scale=smoothing,
                    loc=locs
                ),
                axis=-1
            ) * self.particle_weights,
            axis=1
        ).reshape(p1s.shape) # Finally, reshape back into the same shape as the mesh.
        
        return p1s, p2s, pr
    
    def plot_posterior_contour(self, idx_param1=0, idx_param2=1, res1=100, res2=100, smoothing=0.01):
        """
        Plots a contour of the kernel density estimation
        of a 2D projection of the current posterior distribution.
        
        :param int idx_param1: Parameter to be treated as :math:`x` when
            plotting.
        :param int idx_param2: Parameter to be treated as :math:`y` when
            plotting.
        :param int res1: Resolution along the :math:`x` direction.
        :param int res2: Resolution along the :math:`y` direction.
        :param float smoothing: Standard deviation of the Gaussian kernel
            used to smooth the particle approximation to the current posterior.
            
        .. seealso::
        
            :meth:`SMCUpdater.posterior_mesh`
        """
        return plt.contour(*self.posterior_mesh(idx_param1, idx_param2, res1, res2, smoothing))
        
    ## IPYTHON SUPPORT METHODS ################################################
    
    def _repr_html_(self):
        return r"""
        <strong>{cls_name} for model of type {model}:</strong>
        <table>
            <caption>Current estimated parameters</caption>
            <thead>
                <tr>
                    {parameter_names}
                </tr>
            </thead>
            <tbody>
                <tr>
                    {parameter_values}
                </tr>
            </tbody>
        </table>
        <em>Resample count:</em> {resample_count}
        """.format(
            cls_name=type(self).__name__, # Useful for subclassing.
            model=type(self.model).__name__,
            
            parameter_names="\n".join(
                map("<td>${}$</td>".format, self.model.modelparam_names)
            ),
            
            # TODO: change format string based on number of digits of precision
            #       admitted by the variance.
            parameter_values="\n".join(
                "<td>{}</td>".format(
                    format_uncertainty(mu, std)
                )
                for mu, std in
                zip(self.est_mean(), np.sqrt(np.diag(self.est_covariance_mtx())))
            ),
            
            resample_count=self.resample_count
        )
        
          
class MixedApproximateSMCUpdater(SMCUpdater):
    """
    Subclass of :class:`SMCUpdater` that uses a mixture of two models, one
    of which is assumed to be expensive to compute, while the other is assumed
    to be cheaper. This allows for approximate computation to be used on the
    lower-weight particles.

    :param ~qinfer.abstract_model.Model good_model: The more expensive, but
        complete model.
    :param ~qinfer.abstract_model.Model approximate_model: The less expensive,
        but approximate model.
    :param float mixture_ratio: The ratio of the posterior weight that will
        be delegated to the good model in each update step.
    :param float mixture_thresh: Any particles of weight at least equal to this
        threshold will be delegated to the complete model, irrespective
        of the value of ``mixture_ratio``.
    :param int min_good: Minimum number of "good" particles to assign at each
        step.
        
    All other parameters are as described in the documentation of
    :class:`SMCUpdater`.
    """
                
    def __init__(self,
            good_model, approximate_model,
            n_particles, prior,
            resample_a=None, resampler=None, resample_thresh=0.5,
            mixture_ratio=0.5, mixture_thresh=1.0, min_good=0
            ):
            
        self._good_model = good_model
        self._apx_model = approximate_model
        
        super(MixedApproximateSMCUpdater, self).__init__(
            good_model, n_particles, prior,
            resample_a, resampler, resample_thresh
        )
        
        self._mixture_ratio = mixture_ratio
        self._mixture_thresh = mixture_thresh
        self._min_good = min_good
                
    def hypothetical_update(self, outcomes, expparams, return_likelihood=False, return_normalization=False):
        # TODO: consolidate code with SMCUpdater by breaking update logic
        #       into private method.
        
        # It's "hypothetical", don't want to overwrite old weights yet!
        weights = self.particle_weights
        locs = self.particle_locations

        # Check if we have a single outcome or an array. If we only have one
        # outcome, wrap it in a one-index array.
        if not isinstance(outcomes, np.ndarray):
            outcomes = np.array([outcomes])

        # Make an empty array for likelihoods. We'll fill it in in two steps,
        # the good step and the approximate step.
        L = np.zeros((outcomes.shape[0], locs.shape[0], expparams.shape[0]))

        # Which indices go to good_model?
        
        # Start by getting a permutation that sorts the weights.
        # Since sorting as implemented by NumPy is stable, we want to break
        # that stability to avoid introducing patterns, and so we first
        # randomly shuffle the identity permutation.
        idxs_random = np.arange(weights.shape[0])
        np.random.shuffle(idxs_random)
        idxs_sorted = np.argsort(weights[idxs_random])
        
        # Find the inverse permutation to be that which returns
        # the composed permutation sort º shuffle to the identity.
        inv_idxs_sort = np.argsort(idxs_random[idxs_sorted])
        
        # Now strip off a set of particles producing the desired total weight
        # or that have weights above a given threshold.
        sorted_weights = weights[idxs_random[idxs_sorted]]
        cum_weights = np.cumsum(sorted_weights)
        good_mask = (np.logical_or(
            cum_weights >= 1 - self._mixture_ratio,
            sorted_weights >= self._mixture_thresh
        ))[inv_idxs_sort]
        if np.sum(good_mask) < self._min_good:
            # Just take the last _min_good instead of something sophisticated.
            good_mask = np.zeros_like(good_mask)
            good_mask[idxs_random[idxs_sorted][-self._min_good:]] = True
        bad_mask = np.logical_not(good_mask)
        
        # Finally, separate out the locations that go to each of the good and
        # bad models.
        locs_good = locs[good_mask, :]
        locs_bad = locs[bad_mask, :]
        
        assert_thresh=1e-6
        assert np.mean(weights[good_mask]) - np.mean(weights[bad_mask]) >= -assert_thresh
        
        # Now find L for each of the good and bad models.
        L[:, good_mask, :] = self._good_model.likelihood(outcomes, locs_good, expparams)
        L[:, bad_mask, :] = self._apx_model.likelihood(outcomes, locs_bad, expparams)
        L = L.transpose([0, 2, 1])
        
        # update the weights sans normalization
        # Rearrange so that likelihoods have shape (outcomes, experiments, models).
        # This makes the multiplication with weights (shape (models,)) make sense,
        # since NumPy broadcasting rules align on the right-most index.
        hyp_weights = weights * L
        
        # Sum up the weights to find the renormalization scale.
        norm_scale = np.sum(hyp_weights, axis=2)[..., np.newaxis]
        
        # As a special case, check whether any entries of the norm_scale
        # are zero. If this happens, that implies that all of the weights are
        # zero--- that is, that the hypothicized outcome was impossible.
        # Conditioned on an impossible outcome, all of the weights should be
        # zero. To allow this to happen without causing a NaN to propagate,
        # we forcibly set the norm_scale to 1, so that the weights will
        # all remain zero.
        #
        # We don't actually want to propagate this out to the caller, however,
        # and so we save the "fixed" norm_scale to a new array.
        fixed_norm_scale = norm_scale.copy()
        fixed_norm_scale[np.abs(norm_scale) < np.spacing(0)] = 1
        
        # normalize
        norm_weights = hyp_weights / fixed_norm_scale
            # Note that newaxis is needed to align the two matrices.
            # This introduces a length-1 axis for the particle number,
            # so that the normalization is broadcast over all particles.
        if not return_likelihood:
            if not return_normalization:
                return norm_weights
            else:
                return norm_weights, norm_scale
        else:
            if not return_normalization:
                return norm_weights, L
            else:
                return norm_weights, L, norm_scale
                
class SMCUpdaterBCRB(SMCUpdater):
    """

    Subclass of :class:`SMCUpdater`, adding Bayesian Cramer-Rao bound
    functionality.
    
    Models considered by this class must be differentiable.
    
    In addition to the arguments taken by :class:`SMCUpdater`, this class
    takes the following keyword-only arguments:
        
    :param bool adaptive: If `True`, the updater will track both the
        non-adaptive and adaptive Bayes Information matrices.
    :param initial_bim: If the regularity conditions are not met, then taking
        the outer products of gradients over the prior will not give the correct
        initial BIM. In such cases, ``initial_bim`` can be set to the correct
        BIM corresponding to having done no experiments.
    """
    


    def __init__(self, *args, **kwargs):
        SMCUpdater.__init__(self, *args, **{
            key: kwargs[key] for key in kwargs
            if key in [
                'resampler_a', 'resampler', 'resample_thresh', 'model',
                'prior', 'n_particles'
            ]
        })
        
        if not isinstance(self.model, DifferentiableModel):
            raise ValueError("Model must be differentiable.")
        
        # TODO: fix distributions to make grad_log_pdf return the right
        #       shape, such that the indices are
        #       [idx_model, idx_param] → [idx_model, idx_param],
        #       so that prior.grad_log_pdf(modelparams[i, j])[i, k]
        #       returns the partial derivative with respect to the kth
        #       parameter evaluated at the model parameter vector
        #       modelparams[i, :].
        if 'initial_bim' not in kwargs or kwargs['initial_bim'] is None:
            gradients = self.prior.grad_log_pdf(self.particle_locations)
            self.current_bim = np.sum(
                gradients[:, :, np.newaxis] * gradients[:, np.newaxis, :],
                axis=0
            ) / self.n_particles
        else:
            self.current_bim = kwargs['initial_bim']
            
        # Also track the adaptive BIM, if we've been asked to.
        if "adaptive" in kwargs and kwargs["adaptive"]:
            self._track_adaptive = True
            # Both the prior- and posterior-averaged BIMs start
            # from the prior.
            self.adaptive_bim = self.current_bim.copy()
        else:
            self._track_adaptive = False
    
    # TODO: since we are guaranteed differentiability, and since SMCUpdater is
    #       now a Distribution subclass representing posterior sampling, write
    #       a grad_log_pdf for the posterior distribution, perhaps?
        
    def _bim(self, modelparams, expparams, modelweights=None):
        # TODO: document
        #       rough idea of this function is to take expectations of an
        #       FI over some distribution, here represented by modelparams.
        
        # NOTE: The signature of this function is a bit odd, but it allows
        #       us to represent in one function both prior samples of uniform
        #       weight and weighted samples from a posterior.
        #       Because it's a bit odd, we make it a private method and expose
        #       functionality via the prior_bayes_information and
        #       posterior_bayes_information methods.
        
        # About shapes: the FI we will be averaging over has four indices:
        # FI[i, j, m, e], i and j being matrix indices, m being a model index
        # and e being a model index.
        # We will thus want to return an array of shape BI[i, j, e], reducing
        # over the model index.
        fi = self.model.fisher_information(modelparams, expparams)
        
        # We now either reweight and sum, or sum and divide, based on whether we
        # have model weights to consider or not.
        if modelweights is None:
            # Assume uniform weights.
            bim = np.sum(fi, axis=2) / modelparams.shape[0]
        else:
            bim = np.einsum("m,ijme->ije", modelweights, fi)
            
        return bim
        
    def prior_bayes_information(self, expparams, n_samples=None):
        if n_samples is None:
            n_samples = self.particle_locations.shape[0]
        return self._bim(self.prior.sample(n_samples), expparams)
        
    def posterior_bayes_information(self, expparams):
        return self._bim(
            self.particle_locations, expparams,
            modelweights=self.particle_weights
        )
        
    def update(self, outcome, expparams):
        # Before we update, we need to commit the new Bayesian information
        # matrix corresponding to the measurement we just made.
        self.current_bim += self.prior_bayes_information(expparams)[:, :, 0]
        
        # If we're tracking the information content accessible to adaptive
        # algorithms, then we must use the current posterior as the prior
        # for the next step, then add that accordingly.
        if self._track_adaptive:
            self.adaptive_bim += self.posterior_bayes_information(expparams)[:, :, 0]
        
        # We now can update as normal.
        SMCUpdater.update(self, outcome, expparams)
        

class SMCUpdaterABC(SMCUpdater):
    """

    Subclass of :class:`SMCUpdater`, adding approximate Bayesian computation
    functionality.
    
    """

    def __init__(self, model, n_particles, prior,
                 abc_tol=0.01, abc_sim=1e4, **kwargs):
        self.abc_tol = abc_tol
        self.abc_sim = abc_sim
        
        SMCUpdater.__init__(self, model, n_particles, prior, **kwargs)
        
    def hypothetical_update(self, outcomes, expparams):
        weights = np.copy(self.particle_weights)

        # Check if we have a single outcome or an array. If we only have one
        # outcome, wrap it in a one-index array.
        if not isinstance(outcomes, np.ndarray):
            outcomes = np.array([outcomes])
        
        #TODO: lots of assumptions have been made to ensure the following works
        # 1 - this may only work for binary outcomes
        # 2 - in any case, it assumes the outcome of an experiment is a single number
        
        # first simulate abc_sim experiments
        n = self.model.simulate_experiment(self.particle_locations, expparams, repeat=self.abc_sim)
        # re-weight the particle by multiplying by number of simulated 
        # that came within a tolerance of abc_tol of the actual outcome    
        weights = weights * np.sum(np.abs(n-outcomes)/self.abc_sim <= self.abc_tol,1) 
        
        # normalize
        return weights / np.sum(weights)
        
    def update(self, outcome, expparams, check_for_resample=True):
        self.particle_weights = self.hypothetical_update(outcome, expparams)

        if check_for_resample:
            self._maybe_resample()

