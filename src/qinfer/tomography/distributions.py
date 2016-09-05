#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# distributions.py: Fiducial and informative prior distributions for quantum
#     states and channels.
##
# © 2015 Chris Ferrie (csferrie@gmail.com) and
#        Christopher E. Granade (cgranade@cgranade.com).
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

# TODO: docstrings!
# TODO: unit tests!

## FEATURES ##################################################################

from __future__ import absolute_import
from __future__ import division

## IMPORTS ###################################################################

from qinfer import Distribution, SingleSampleMixin
from qinfer.tomography.bases import gell_mann_basis, tensor_product_basis

import abc
import itertools as it

import numpy as np

# Since the rest of QInfer does not require QuTiP,
# we need to import it in a way that we don't propagate exceptions if QuTiP
# is missing or is too early a version.
from qinfer.utils import get_qutip_module
qt = get_qutip_module('3.2')

import warnings

## EXPORTS ###################################################################

__all__ = [
    'DensityOperatorDistribution',
    'GinibreDistribution',
    'GinibreReditDistribution',
    'BCSZChoiDistribution',
    'GADFLIDistribution',
    'TensorProductDistribution'
]

## FUNCTIONS #################################################################
# TODO: almost all of these bases need moved out, contributed to QuTiP.

def rand_dm_ginibre_redit(N=2, rank=None, dims=None):
    # TODO: contribute to QuTiP!
    if rank is None:
        rank = N
    X = np.random.randn(N * rank).reshape((N, rank))
    rho = np.dot(X, X.T)
    rho /= np.trace(rho)

    return qt.Qobj(rho, dims=dims)

## CLASSES ###################################################################

class DensityOperatorDistribution(SingleSampleMixin, Distribution):
    """
    Distribution over density operators parameterized in a given
    basis.

    :type basis: `int` or :class:`TomographyBasis`
    :param basis: Basis to use in representing sampled
        density operators. If an `int`, assumes a default
        (Gell-Mann) basis of that dimension.
    """
    def __init__(self, basis):
        if isinstance(basis, int):
            basis = gell_mann_basis(basis)

        self._dim = basis.dim
        self._basis = basis

    @abc.abstractmethod
    def _sample_dm(self):
        pass

    @property
    def n_rvs(self):
        """
        Number of random variables represented by this distribution.
        
        :type: `int`
        """
        return self._dim **2    

    @property
    def dim(self):
        """
        Dimension of the Hilbert space on which sampled density operators
        act.

        :type: `int`
        """
        return self._dim

    @property
    def basis(self):
        """
        Basis used to represent sampled density operators as model parameter
        vectors.
        """
        return self._basis

    def _sample(self):
        sample_dm = self._sample_dm()
        sample_dm /= sample_dm.tr()
        return self.basis.state_to_modelparams(sample_dm)

class TensorProductDistribution(DensityOperatorDistribution):
    """
    This class is implemented using QuTiP (v3.1.0 or later), and thus will not
    work unless QuTiP is installed.
    
    :param factors: Distributions representing each factor of the tensor
        product used to generate samples.
    :type factors: `list` of :class:`DensityOperatorDistribution`
        instances
    """
    
    def __init__(self, factors):
        super(TensorProductDistribution, self).__init__(
            basis=tensor_product_basis(
                factor.basis for factor in factors
            )
        )
        self._factors = tuple(factors)

    def _sample_dm(self):
        return qt.tensor([
            factor_dist._sample_dm() for factor_dist in self._factors
        ])


class GinibreDistribution(DensityOperatorDistribution):
    """
    Distribution over all trace-1 positive semidefinite operators
    of a given rank. Generalizes the Hilbert-Schmidt
    (full-rank) and Haar (rank-1) distributions.

    :param TomographyBasis basis: Basis to use in generating
        samples.
    :param int rank: Rank of each sampled state. If `None`,
        defaults to full-rank.
    """
    
    def __init__(self, basis, rank=None):
        super(GinibreDistribution, self).__init__(basis)
        if rank is not None and rank > self.dim:
            raise ValueError("rank must not exceed basis dimension.")
        self._rank = rank

    def __repr__(self):
        return "<GinibreDistribution dims={}, rank={}, basis={}>".format(
            self.dim,
            self._rank if self._rank is not None else self.dim,
            self.basis.name
        )

    def _sample_dm(self):
        # Generate and flatten a density operator, so that we can multiply it
        # by the transformation defined above.
        return qt.rand_dm_ginibre(self._dim, rank=self._rank)

class GinibreReditDistribution(DensityOperatorDistribution):
    """
    Distribution over all real-valued trace-1 positive semidefinite
    operators of a given rank. Generalizes the Hilbert-Schmidt
    (full-rank) and Haar (rank-1) distributions. Useful for plotting.

    :param TomographyBasis basis: Basis to use in generating
        samples.
    :param int rank: Rank of each sampled state. If `None`,
        defaults to full-rank.
    """

    def __init__(self, basis, rank=None):
        super(GinibreReditDistribution, self).__init__(basis)
        self._rank = rank

    def _sample_dm(self):
        # Generate and flatten a density operator, so that we can multiply it
        # by the transformation defined above.
        return rand_dm_ginibre_redit(self._dim, rank=self._rank)

class BCSZChoiDistribution(DensityOperatorDistribution):
    """
    Samples Choi states for completely-positive (CP) or CP and
    trace-preserving (CPTP) maps, as generated
    by the BCSZ prior [BCSZ09]_. The sampled states are normalized
    as states (trace 1).
    """
    def __init__(self, basis, rank=None, enforce_tp=True):
        if isinstance(basis, int):
            basis = gell_mann_basis(basis)
        self._hdim = basis.dim

        # TODO: take basis on underlying space, tensor up?
        channel_basis = tensor_product_basis(basis, basis)
        # FIXME: this is a hack to get another level of nesting.
        channel_basis.dims = [basis.dims, basis.dims]
        channel_basis.superrep = 'choi'
        super(BCSZChoiDistribution, self).__init__(channel_basis)
        self._rank = rank
        self._enforce_tp = enforce_tp

    def _sample_dm(self):
        return qt.to_choi(
            qt.rand_super_bcsz(self._hdim, self._enforce_tp, self._rank)
        ).unit()

class GADFLIDistribution(DensityOperatorDistribution):
    """
    Samples operators from the generalized amplitude damping prior
    for liklihood-based inference [GCC16]_, given a fiducial
    distribution and the desired mean for the prior.

    :param DensityOperatorDistribution fiducial_distribution:
        Distribution from which samples are initially drawn
        before transformation under generalized amplitude damping.
    :param qutip.Qobj mean: State which will be the mean of the
        GAD-transformed samples.
    """
    def __init__(self, fiducial_distribution, mean):
        super(GADFLIDistribution, self).__init__(fiducial_distribution.basis)
        self._fid = fiducial_distribution
        mean = (
            qt.to_choi(mean).unit()
            if mean.type == 'super' and not mean.superrep == 'choi' else
            mean
        )
        self._mean = mean

        alpha = 1
        lambda_min = min(mean.eigenenergies())
        if lambda_min < 0:
            raise ValueError("Negative eigenvalue {} in informative mean.".format(lambda_min))
        d = self.dim
        beta = (
            1 / (d * lambda_min - 1) - 1
        ) if lambda_min > 0.5 else (
            (d * lambda_min) / (1 - d * lambda_min)
        )
        if beta < 0:
            raise ValueError("Beta < 0 for informative mean.")
        self._alpha = alpha
        self._beta = beta

        eye = qt.qeye(self._dim).unit()
        eye.dims = mean.dims
        self._rho_star = (alpha + beta) / alpha * (
            mean - (beta) / (alpha + beta) * eye.unit()
        )

    def _sample_dm(self):
        fid_samp = self._fid._sample_dm()
        eps = np.random.beta(self._alpha, self._beta)
        return (1 - eps) * fid_samp + eps * self._rho_star

