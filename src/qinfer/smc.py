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

## FEATURES ####################################################################

from __future__ import division

## ALL #########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
    'SMCUpdater'
]

## IMPORTS #####################################################################

import numpy as np
import warnings

from abstract_model import DifferentiableModel
from metrics import rescaled_distance_mtx

from resamplers import LiuWestResampler

from scipy.spatial import Delaunay
import scipy.linalg as la
from utils import outer_product, mvee, uniquify, particle_meanfn, \
        particle_covariance_mtx
from _exceptions import ApproximationWarning
import scipy.stats

## CLASSES #####################################################################

class SMCUpdater(object):
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

    ## PROPERTIES ##############################################################

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

    ## PRIVATE METHODS #########################################################
    
    def _maybe_resample(self):
        """
        Checks the resample threshold and conditionally resamples.
        """
        if self.n_ess < self.n_particles * self.resample_thresh:
            self.resample()
            pass

    ## UPDATE METHODS ##########################################################

    def hypothetical_update(self, outcomes, expparams, return_likelihood=False):
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
        weights = np.copy(self.particle_weights)
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
        weights = weights * L
        
        # Sum up the weights to find the renormalization scale.
        norm_scale = np.sum(weights, axis=2)[..., np.newaxis]
        
        # As a special case, check whether any entries of the norm_scale
        # are zero. If this happens, that implies that all of the weights are
        # zero--- that is, that the hypothicized outcome was impossible.
        # Conditioned on an impossible outcome, all of the weights should be
        # zero. To allow this to happen without causing a NaN to propagate,
        # we forcibly set the norm_scale to 1, so that the weights will
        # all remain zero.
        norm_scale[np.abs(norm_scale) < np.spacing(0)] = 1
        
        # normalize
        norm_weights = weights / norm_scale
            # Note that newaxis is needed to align the two matrices.
            # This introduces a length-1 axis for the particle number,
            # so that the normalization is broadcast over all particles.
        if not return_likelihood:
            return norm_weights
        else:
            return norm_weights, L

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

        # Since hypothetical_update returns an array indexed by
        # [outcome, experiment, particle], we need to strip off those two
        # indices first.
        self.particle_weights = self.hypothetical_update(outcome, expparams)[0, 0, :]

        if check_for_resample:
            self._maybe_resample()
            
        assert np.all(self.particle_weights >= 0)

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

    ## RESAMPLING METHODS ######################################################

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


    ## ESTIMATION METHODS ######################################################

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

    def est_covariance_mtx(self):
        """
        Returns an estimate of the covariance of the current posterior model
        distribution, given by the covariance of the current SMC approximation.
        
        :rtype: :class:`numpy.ndarray`, shape
            ``(n_modelparams, n_modelparams)``.
        :returns: An array containing the estimated covariance matrix.
        """
        # Find the mean model vector, shape (n_modelparams, ).
        mu = self.est_mean()
        # Transpose the particle locations to have shape
        # (n_modelparams, n_particles).
        xs = self.particle_locations.transpose([1, 0])
        # Give a shorter name to the particle weights, shape (n_particles, ).
        ws = self.particle_weights

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
        if not np.all(la.eig(cov)[0] >= 0):
            warnings.warn('Numerical error in covariance estimation causing positive semidefinite violation.', ApproximationWarning)

        return cov

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
        
        mu = np.sum(
            # We need the particle index to be the rightmost index, so that
            # the two arrays align on the particle index as opposed to the
            # modelparam index.
            w[:, np.newaxis, :] * xs[np.newaxis, ...],
            # The argument now has shape (n_outcomes, n_modelparams, n_particles),
            # so that the sum should collapse the particle index, 2.
            axis=2
        ) # <- has shape (n_outcomes, n_np).
        
        var = (
            # This sum is a reduction over the particle index, chosen to be
            # axis=2. Thus, the sum represents an expectation value over the
            # outer product $x . x^T$.

            np.sum(w[:, np.newaxis, :] * xs[np.newaxis, ...]**2,
                    axis=2
                ) # <- has shape (n_outcomes, n_mp).
                # We finish by subracting from the above expectation value
                # the outer product $mu . mu^T$.
                - mu**2)


        rescale_var = np.sum(self.model.Q * var, axis=1)
        # Q has shape (n_mp,),
        # therefore <- has shape (n_outcomes,)
        tot_like = np.sum(L, axis=1)
        return np.dot(tot_like.T, rescale_var)
        
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
        
    def est_cluster_moments(self, cluster_opts=None):
        # TODO: document
        
        if cluster_opts is None:
            cluster_opts = {}
        
        for cluster_label, cluster_particles in clustering.cluster(
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
        # TODO: document
        wcv, bcv, tv = self.est_cluster_covs(cluster_opts)
        return wcv / tv
        

    ## REGION ESTIMATION METHODS ###############################################

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
        
    ## MISC METHODS ############################################################
    
    def risk(self, x0):
        return self.bayes_risk(np.array([(x0,)], dtype=self.model.expparams_dtype))
        
                
class SMCUpdaterBCRB(SMCUpdater):
    """

    Subclass of :class:`SMCUpdater`, adding Bayesian Cramer-Rao bound
    functionality.
    
    Models considered by this class must be differentiable.
    """
    


    def __init__(self, *args, **kwargs):
        SMCUpdater.__init__(self, *args, **kwargs)
        
        if not isinstance(self.model, DifferentiableModel):
            raise ValueError("Model must be differentiable.")
        
        self.current_bim = np.sum(np.array([
            outer_product(self.prior.grad_log_pdf(particle))
            for particle in self.particle_locations
            ]), axis=0) / self.n_particles
        
    def hypothetical_bim(self, expparams):
        # E_{prior} E_{data | model, exp} [outer-product of grad-log-likelihood]
        like_bim = np.zeros(self.current_bim.shape)
        
        for idx_particle in xrange(self.n_particles):
        
            modelparams = self.prior.sample()

            weight = 1 / self.n_particles
            
            for outcome in np.arange(self.model.n_outcomes(expparams))[...,np.newaxis]:
                 
                grad = outer_product(self.model.grad_log_likelihood(outcome, modelparams, expparams)) 
                L = self.model.likelihood(outcome, modelparams, expparams)[0]
                like_bim += weight * grad * L
                
        return self.current_bim + like_bim
        
        
    def update(self, outcome, expparams):
        # Before we update, we need to commit the new Bayesian information
        # matrix corresponding to the measurement we just made.
        self.current_bim = self.hypothetical_bim(expparams)
        
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
