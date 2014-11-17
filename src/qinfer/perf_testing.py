#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# perf_testing.py: Tests the performance of SMC estimation and likelihood
#     calls.
##
# © 2014 Chris Ferrie (csferrie@gmail.com) and
#        Christopher E. Granade (cgranade@gmail.com).
#
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

## FEATURES ##################################################################

from __future__ import division

## EXPORTS ###################################################################

__all__ = [
    'timing', 'perf_test'
]

## IMPORTS ###################################################################

from contextlib import contextmanager
import time

import numpy as np

from qinfer.smc import SMCUpdater

## CLASSES ###################################################################

class Timer(object):
    """
    Represents the timing of a block. Call ``stop()`` to stop the
    timer, and query the ``delta_t`` property for the time since this object
    was created.
    """
    _tic = 0
    _toc = None

    def __init__(self):
        self._tic = time.time()

    def stop(self):
        self._toc = time.time()

    @property
    def delta_t(self):
        """
        Returns the time (in seconds) elapsed during the block that was 
        """
        return (self._toc if self._toc is not None else time.time()) - self._tic

## CONTEXT MANAGERS ##########################################################

@contextmanager
def timing():
    """
    Times the execution of a block, returning the result as a
    :class:`qinfer.Timer()`. For example::

    >>> with timing() as t:
    ...     time.sleep(1)
    >>> print t.delta_t # Should return approximately 1.
    """
    t = Timer()
    yield t
    t.stop()

## CONSTANTS #################################################################

PERFORMANCE_DTYPE = [
    ('loss', float),
    ('resample_count', int),
    ('elapsed_time', float),
    ('outcome', int)
]

## FUNCTIONS #################################################################

def perf_test(
        model, n_particles, prior, n_exp, heuristic_class,
        true_model=None, true_prior=None, true_mps = None
    ):
    """
    Runs a trial of using SMC to estimate the parameters of a model, given a
    number of particles, a prior distribution and an experiment design
    heuristic.

    :param qinfer.Model model: Model whose parameters are to
        be estimated.
    :param int n_particles: Number of SMC particles to use.
    :param qinfer.Distribution prior: Prior to use in selecting
        SMC particles.
    :param int n_exp: Number of experimental data points to draw from the
        model.
    :param qinfer.Heuristic heuristic_class: Constructor function
        for the experiment design heuristic to be used.
    :param qinfer.Model true_model: Model to be used in
        generating experimental data. If ``None``, assumed to be ``model``.
    :param qinfer.Distribution true_prior: Prior to be used in
        selecting the true model parameters. If ``None``, assumed to be
        ``prior``.
    :param np.ndarray true_mps: The true model parameters. If ``None``,
        it will be sampled from ``true_prior``. Note that the performance
        record can only handle one outcome and therefore ONLY ONE TRUE MODEL.
        An error will occur if ``true_mps.shape[0] > 1`` returns ``True``.
    :rtype np.ndarray: See :ref:`perf_testing_struct` for more details on 
        the type returned by this function.
    :return: A record array of performance metrics, indexed by the number
        of experiments performed.
    """

    if true_model is None:
        true_model = model

    if true_prior is None:
        true_prior = prior

    if true_mps is None:
        true_mps = true_prior.sample()

    performance = np.zeros((n_exp,), dtype = PERFORMANCE_DTYPE + model.expparams_dtype)

    updater = SMCUpdater(model, n_particles, prior)
    heuristic = heuristic_class(updater)

    for idx_exp in xrange(n_exp):
        expparams = heuristic()
        datum = true_model.simulate_experiment(true_mps, expparams)

        with timing() as t:
            updater.update(datum, expparams)

        delta = updater.est_mean() - true_mps

        performance[idx_exp]['elapsed_time'] = t.delta_t
        performance[idx_exp]['loss'] = np.dot(delta**2, model.Q)
        performance[idx_exp]['resample_count'] = updater.resample_count
        performance[idx_exp]['outcome'] = datum
        for param_name in [param[0] for param in model.expparams_dtype]:
            performance[idx_exp][param_name] = expparams[param_name]

    return performance
