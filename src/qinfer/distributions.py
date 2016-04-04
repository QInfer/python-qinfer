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

from functools import partial

import abc

from qinfer import utils as u

import warnings

## EXPORTS ###################################################################

__all__ = [
    'Distribution',
    'SingleSampleMixin',
    
    'ProductDistribution',
    'UniformDistribution',
    'ConstantDistribution',
    'NormalDistribution',
    'MultivariateNormalDistribution',
    'SlantedNormalDistribution',
    'LogNormalDistribution',
    'GinibreUniform',
    'HaarUniform',
    'HilbertSchmidtUniform',
    'PostselectedDistribution'
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
        
        :rtype: `int`
        """
        pass
    
    @abc.abstractmethod
    def sample(self, n=1):
        """
        Returns one or more samples from this probability distribution.
        
        :param int n: Number of samples to return.
        :return numpy.ndarray: An array containing samples from the
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

class ProductDistribution(Distribution):
    r"""
    Takes a non-zero number of QInfer distributions :math:`D_k` as input
    and returns their Cartesian product.
    
    In other words, the returned distribution is
    :math:`\Pr(\prod_k D_k) = \prod_k \Pr(D_k)`.
    
    :param *factors: Distribution objects representing :math:`D_k`.
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

class UniformDistributionWith0(Distribution):
    """
    Uniform distribution on a given rectangular region with padded zeros.
    
    :param numpy.ndarray ranges: Array of shape ``(n_rvs, 2)``, where ``n_rvs``
        is the number of random variables, specifying the upper and lower limits
        for each variable.
    """
    
    def __init__(self, ranges=_DEFAULT_RANGES, zeros = 0):
        warnings.warn(
            "This class has been superceded by ProductDistribution and ConstantDistribution.",
            DeprecationWarning
        )
    
        if not isinstance(ranges, np.ndarray):
            ranges = np.array(ranges)
            
        if len(ranges.shape) == 1:
            ranges = ranges[np.newaxis, ...]
        
        self._ranges = ranges
        self._n_rvs = ranges.shape[0]
        self._delta = ranges[:, 1] - ranges[:, 0]
        
        self.zeros = zeros
        
    @property
    def n_rvs(self):
        return self._n_rvs
        
    def sample(self, n=1):
        shape = (n, self._n_rvs)# if n == 1 else (self._n_rvs, n)
        z = np.random.random(shape)
        foo =  self._ranges[:, 0] + z * self._delta
        return np.pad(foo,((0,0), (0, self.zeros)), mode = 'constant')
        
    def grad_log_pdf(self, var):
        # THIS IS NOT TECHNICALLY LEGIT; BCRB doesn't technically work with a
        # prior that doesn't go to 0 at its end points.  But we do it anyway.
        if var.shape[0] == 1:
            return 12/(self._delta)**2
        else:
            return np.zeros(var.shape)


class NormalDistribution(Distribution):
    """

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
        
        return -np.dot(self.invcov,(x - self.mean).transpose()).transpose()
        
        
class SlantedNormalDistribution(Distribution):
    """
    Uniform distribution on a given rectangular region.
    
    :param numpy.ndarray ranges: Array of shape ``(n_rvs, 2)``, where ``n_rvs``
        is the number of random variables, specifying the upper and lower limits
        for each variable.
    """
    
    def __init__(self, ranges=_DEFAULT_RANGES, weight=0.01):
        if not isinstance(ranges, np.ndarray):
            ranges = np.array(ranges)
            
        if len(ranges.shape) == 1:
            ranges = ranges[np.newaxis, ...]
    
        self._ranges = ranges
        self._n_rvs = ranges.shape[0]
        #self._delta = ranges[:, 1] - ranges[:, 0]
        self._weight = weight
        
        
        
    @property
    def n_rvs(self):
        return self._n_rvs
        
    def sample(self, n=1):
        shape = (n, self._n_rvs)# if n == 1 else (self._n_rvs, n)
        z = np.random.randn(n,self._n_rvs)
        return self._ranges[:, 0] +self._weight*z+np.random.rand(n)*self._ranges[:, 1];

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

class MVUniformDistribution(object):
    
    def __init__(self, dim = 6):
        self.dim = dim
                    
    def sample(self, n = 1):
        return np.random.mtrand.dirichlet(np.ones(self.dim),n)

class DiscreteUniformDistribution(Distribution):
    def __init__(self, num_bits):
        self._num_bits = num_bits

    @property
    def n_rvs(self):
        return 1
        
    def sample(self, n=1):
        z = np.random.randint(2**self._num_bits,n)
        return z
    
class HilbertSchmidtUniform(SingleSampleMixin, Distribution):
    """
    Creates a new Hilber-Schmidt uniform prior on state space of dimension ``dim``.
    See e.g. [Mez06]_ and [Mis12]_.

    :param int dim: Dimension of the state space.
    """
    def __init__(self, dim=2):
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
    Creates a new Haar uniform prior on state space of dimension ``dim``.

    :param int dim: Dimension of the state space.
    """
    def __init__(self, dim=2):
        self.dim = dim

    @property
    def n_rvs(self):
        return 3    
    
    def sample(self):
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
        self.dim = dim
        self.k = k

    @property
    def n_rvs(self):
        return 3        
        
    def sample(self):
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
