#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# plotting_tools.py: Functions for plotting tomographic data and estimates.
##
# © 2015 Chris Ferrie (csferrie@gmail.com) and
#        Christopher E. Granade (cgranade@cgranade.com),
#        except where otherwise noted.
# Based on work with Joshua Combes (joshua.combes@gmail.com).
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

# TODO: unit tests!

## FEATURES ##################################################################

from __future__ import absolute_import
from __future__ import division

## IMPORTS ###################################################################

from builtins import map

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Polygon
except ImportError:
    import warnings
    warnings.warn("Could not import matplotlib.")
    plt = None
    Ellipse = None

# Since the rest of QInfer does not require QuTiP,
# we need to import it in a way that we don't propagate exceptions if QuTiP
# is missing or is too early a version.
from qinfer.utils import get_qutip_module
qt = get_qutip_module('3.2')

## EXPORTS ###################################################################

__all__ = [
    'plot_rebit_modelparams',
    'plot_decorate_rebits',
    'plot_cov_ellipse',
    'plot_rebit_prior',
    'plot_rebit_posterior'
]

## CONSTANTS #################################################################

REBIT_AXES = [1, 2]

## FUNCTIONS #################################################################

def plot_rebit_modelparams(modelparams, rebit_axes=REBIT_AXES, **kwargs):
    """
    Given model parameters representing rebits, plots the
    rebit states as a scatter plot. Additional keyword arguments
    are passed to :ref:`plt.scatter`.

    :param np.ndarray modelparams: Model parameters representing
        rebits.
    :param list rebit_axes: List containing indices for the :math:`x`
        and :math:`z` axes.
    """
    mps = modelparams[:, rebit_axes] * np.sqrt(2)
    plt.scatter(mps[:, 0], mps[:, 1], **kwargs)

def plot_decorate_rebits(basis=None, rebit_axes=REBIT_AXES):
    """
    Decorates a figure with the boundary of rebit state space
    and basis labels drawn from a :ref:`~qinfer.tomography.TomographyBasis`.

    :param qinfer.tomography.TomographyBasis basis: Basis to use in
        labeling axes.
    :param list rebit_axes: List containing indices for the :math:`x`
        and :math:`z` axes.
    """
    ax = plt.gca()

    if basis is not None:
        labels = list(map(r'$\langle\!\langle {} | \rho \rangle\!\rangle$'.format,
            # Pick out the x and z by default.
            [basis.labels[rebit_axes[0]], basis.labels[rebit_axes[1]]]
        ))
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])

    ax.add_artist(plt.Circle([0, 0], 1, color='k', fill=False))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    # Copied from https://github.com/joferkington/oost_paper_code in
    # accordance with its license agreement.
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    :param cov: The 2x2 covariance matrix to base the ellipse on.
    :param pos: The location of the center of the ellipse. Expects a 2-element
        sequence of ``[x0, y0]``.
    :param nstd: The radius of the ellipse in numbers of standard deviations.
        Defaults to 2 standard deviations.
    :param ax: The axis that the ellipse will be plotted on. Defaults to the 
        current axis.

    :return: A matplotlib ellipse artist.
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


def plot_rebit_prior(prior, rebit_axes=REBIT_AXES,
        n_samples=2000, true_state=None, true_size=250,
        force_mean=None,
        legend=True,
        mean_color_index=2
    ):
    """
    Plots rebit states drawn from a given prior.

    :param qinfer.tomography.DensityOperatorDistribution prior: Distribution over
        rebit states to plot.
    :param list rebit_axes: List containing indices for the :math:`x`
        and :math:`z` axes.
    :param int n_samples: Number of samples to draw from the
        prior.
    :param np.ndarray true_state: State to be plotted as a "true" state for
        comparison.
    """
    pallette = plt.rcParams['axes.color_cycle']

    plot_rebit_modelparams(prior.sample(n_samples),
        c=pallette[0],
        label='Prior',
        rebit_axes=rebit_axes
    )

    if true_state is not None:
        plot_rebit_modelparams(true_state,
            c=pallette[1],
            label='True', marker='*', s=true_size,
            rebit_axes=rebit_axes
        )

    if hasattr(prior, '_mean') or force_mean is not None:
        mean = force_mean if force_mean is not None else prior._mean
        plot_rebit_modelparams(
            prior._basis.state_to_modelparams(mean)[None, :],
            edgecolors=pallette[mean_color_index], s=250, facecolors='none', linewidth=3,
            label='Mean',
            rebit_axes=rebit_axes
        )

    plot_decorate_rebits(prior.basis,
        rebit_axes=rebit_axes
    )
    if legend:
        plt.legend(loc='lower left', ncol=3, scatterpoints=1)


def plot_rebit_posterior(updater, prior=None, true_state=None, n_std=3, rebit_axes=REBIT_AXES, true_size=250,
            legend=True,
            level=0.95,
            region_est_method='cov'
    ):
    """
    Plots posterior distributions over rebits, including covariance ellipsoids

    :param qinfer.smc.SMCUpdater updater: Posterior distribution over rebits.
    :param qinfer.tomography.DensityOperatorDistribution: Prior distribution
        over rebit states.
    :param np.ndarray true_state: Model parameters for "true" state to plot
        as comparison.
    :param float n_std: Number of standard deviations out from the mean
        at which to draw the covariance ellipse. Only used if
        region_est_method is ``'cov'``.
    :param float level: Credibility level to use for computing
        region estimators from convex hulls.
    :param list rebit_axes: List containing indices for the :math:`x`
        and :math:`z` axes.
    :param str region_est_method: Method to use to draw region estimation.
        Must be one of None, ``'cov'`` or ``'hull'``.
    """
    pallette = plt.rcParams['axes.color_cycle']

    plot_rebit_modelparams(updater.particle_locations,
        c=pallette[0],
        label='Posterior',
        s=12 * np.sqrt(updater.particle_weights * len(updater.particle_weights)),
        rebit_axes=rebit_axes,
        zorder=-10
    )

    plot_rebit_modelparams(true_state,
        c=pallette[1],
        label='True', marker='*', s=true_size,
        rebit_axes=rebit_axes
    )

    if prior is not None:
        plot_rebit_modelparams(
            prior._basis.state_to_modelparams(prior._mean)[None, :],
            edgecolors=pallette[3], s=250, facecolors='none', linewidth=3,
            label='Prior Mean',
            rebit_axes=rebit_axes
        )
    plot_rebit_modelparams(
        updater.est_mean()[None, :],
        edgecolors=pallette[2], s=250, facecolors='none', linewidth=3,
        label='Posterior Mean',
        rebit_axes=rebit_axes
    )

    if region_est_method == 'cov':
        # Multiplying by sqrt{2} to rescale to Bloch ball.
        cov = 2 * updater.est_covariance_mtx()
        # Use fancy indexing to cut out all but the desired submatrix.
        cov = cov[rebit_axes, :][:, rebit_axes]
        plot_cov_ellipse(
            cov, updater.est_mean()[rebit_axes] * np.sqrt(2),
            nstd=n_std,
            edgecolor='k', fill=True, lw=2,
            facecolor=pallette[0],
            alpha=0.4,
            zorder=-9,
            label='Posterior Cov Ellipse ($Z = {}$)'.format(n_std)
        )

    elif region_est_method == 'hull':
        # Find the convex hull from the updater, projected
        # on the rebit axes.
        faces, vertices = updater.region_est_hull(level, modelparam_slice=rebit_axes)
        polygon = Polygon(vertices * np.sqrt(2),
            facecolor=pallette[0], alpha=0.4, zorder=-9,
            label=r'Credible Region ($\alpha = {}$)'.format(level),
            edgecolor='k', lw=2, fill=True
        )
        # TODO: consolidate add_patch code with that above.
        plt.gca().add_patch(polygon)

        
    plot_decorate_rebits(updater.model.base_model._basis,
        rebit_axes=rebit_axes
    )

    if legend:
        plt.legend(loc='lower left', ncol=4, scatterpoints=1)

