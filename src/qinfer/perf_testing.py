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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

## EXPORTS ###################################################################

__all__ = [
    'timing', 'perf_test', 'perf_test_multiple'
]

## IMPORTS ###################################################################

from builtins import range

from contextlib import contextmanager
from functools import partial
import time

import numpy as np
import numpy.ma as ma

from qinfer.smc import SMCUpdater
from qinfer.utils import pretty_time

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

    def __repr__(self):
        return "<qinfer.Timer at 0x{0:x}, {1} elapsed>".format(
            id(self), pretty_time(self.delta_t)
        )

    def __str__(self):
        return "{0} elapsed".format(
            pretty_time(self.delta_t)
        )

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


@contextmanager
def numpy_err_policy(**kwargs):
    """
    Uses :ref:`np.seterr` to set an error policy for NumPy functions
    called during the context manager block. For example::

    >>> with numpy_err_policy(divide='raise'):
    ...     # NumPy divsion errors here are exceptions.
    >>> # NumPy division errors here follow the previous policy.
    """

    old_errs = np.seterr(**kwargs)
    yield
    np.seterr(**old_errs)

## CONSTANTS #################################################################

PERFORMANCE_DTYPE = [
    ('loss', float),
    ('resample_count', int),
    ('elapsed_time', float),
    ('outcome', int)
]

## FUNCTIONS #################################################################

def actual_dtype(model, true_model=None):
    if true_model is None:
        true_model = model

    model_dtype = [
        ('true', float, true_model.n_modelparams),
        ('est',  float, model.n_modelparams),
    ]
    if isinstance(model.expparams_dtype, str):
        # They're using simple notation for a single field.
        return PERFORMANCE_DTYPE + model_dtype + [('experiment', model.expparams_dtype)], True
    else:
        return PERFORMANCE_DTYPE + model_dtype + model.expparams_dtype, False

def promote_dims_left(arr, ndim):
    if np.ndim(arr) < ndim:
        return arr[(None,) * (ndim - np.ndim(arr)) + (Ellipsis, )]
    else:
        return arr

def shorten_right(*args):
    # First, ensure that all args have the same number of
    # dims.
    max_dims = max(np.ndim(arg) for arg in args)
    args = list(map(partial(promote_dims_left, ndim=max_dims), args))

    # Next, for each axis, find the *shortest* along that axis.
    min_shapes = [
        min(np.shape(arg)[axis] for arg in args)
        for axis in range(max_dims)
    ]

    # We then trim the elements that are longer than the minimum shape.
    min_slice = np.s_[tuple([
        np.s_[-min_shape:]
        for min_shape in min_shapes
    ])]
    return tuple([
        arg[min_slice] for arg in args
    ])

def perf_test(
        model, n_particles, prior, n_exp, heuristic_class,
        true_model=None, true_prior=None, true_mps=None,
        extra_updater_args=None
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
        Note that if the true and estimation models have different numbers
        of parameters, the loss will be calculated by aligning the
        respective model vectors "at the right," analogously to the
        convention used by NumPy broadcasting.
    :param qinfer.Distribution true_prior: Prior to be used in
        selecting the true model parameters. If ``None``, assumed to be
        ``prior``.
    :param numpy.ndarray true_mps: The true model parameters. If ``None``,
        it will be sampled from ``true_prior``. Note that as this function
        runs exactly one trial, only one model parameter vector may be passed.
        In particular, this requires that ``len(true_mps.shape) == 1``. 
    :param dict extra_updater_args: Extra keyword arguments for the updater,
        such as resampling and zero-weight policies.
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

    if extra_updater_args is None:
        extra_updater_args = {}

    n_min_modelparams = min(model.n_modelparams, true_model.n_modelparams)

    dtype, is_scalar_exp = actual_dtype(model, true_model)
    performance = np.zeros((n_exp,), dtype=dtype)

    updater = SMCUpdater(model, n_particles, prior, **extra_updater_args)
    heuristic = heuristic_class(updater)

    for idx_exp in range(n_exp):
        # Set inside the loop to handle the case where the
        # true model is time-dependent as well as the estimation model.
        performance[idx_exp]['true'] = true_mps

        expparams = heuristic()
        datum = true_model.simulate_experiment(true_mps, expparams)

        with timing() as t:
            updater.update(datum, expparams)

        # Update the true model.
        true_mps = true_model.update_timestep(
            promote_dims_left(true_mps, 2), expparams
        )[:, :, 0]

        est_mean = updater.est_mean()
        delta = np.subtract(*shorten_right(est_mean, true_mps))
        loss = np.dot(delta**2, model.Q[-n_min_modelparams:])

        performance[idx_exp]['elapsed_time'] = t.delta_t
        performance[idx_exp]['loss'] = loss
        performance[idx_exp]['resample_count'] = updater.resample_count
        performance[idx_exp]['outcome'] = datum
        performance[idx_exp]['est'] = est_mean
        if is_scalar_exp:
            performance[idx_exp]['experiment'] = expparams
        else:
            for param_name in [param[0] for param in model.expparams_dtype]:
                performance[idx_exp][param_name] = expparams[param_name]

    return performance


class apply_serial(object):
    """
    Applies the function ``fn`` in the main thread. Used
    to emulate the API exposed by parallelization engines.
    """
    _value = None
    _done = False

    def __init__(self, fn, *args, **kwargs):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
    
    def get(self):
        if not self._done:
            self._value = self._fn(*self._args, **self._kwargs)
            self._done = True

        return self._value

def perf_test_multiple(
        n_trials,
        model, n_particles, prior,
        n_exp, heuristic_class,
        true_model=None, true_prior=None, true_mps=None,
        apply=apply_serial,
        allow_failures=False,
        extra_updater_args=None,
        progressbar=None
    ):
    """
    Runs many trials of using SMC to estimate the parameters of a model, given a
    number of particles, a prior distribution and an experiment design
    heuristic.

    In addition to the parameters accepted by :func:`perf_test`,
    this function takes the following arguments:

    :param int n_trials: Number of different trials to run.
    :param callable apply: Function to call to delegate each trial.
        See, for example, :meth:`~ipyparallel.LoadBalancedView.apply`.
    :param qutip.ui.BaseProgressBar progressbar: QuTiP-style progress bar
        class used to report how many trials have successfully completed.
    :param bool allow_failures: If `False`, an exception raised
        in any trial will propagate out. Otherwise, failed
        trials are masked out of the returned performance array
        using NumPy masked arrays.
    :rtype np.ndarray: See :ref:`perf_testing_struct` for more details on 
        the type returned by this function.
    :return: A record array of performance metrics, indexed by
        the trial and the number of experiments performed.
    """

    trial_fn = partial(perf_test,
        model, n_particles, prior,
        n_exp, heuristic_class, true_model, true_prior,
        true_mps=true_mps,
        extra_updater_args=extra_updater_args
    )

    dtype, is_scalar_exp = actual_dtype(model, true_model)
    performance = (np.zeros if not allow_failures else ma.zeros)((n_trials, n_exp), dtype=dtype)

    prog = None

    try:
        name = getattr(type(model), '__name__', 'unknown model')
    except:
        name = 'unknown model'

    try:
        if progressbar is not None:
            prog = progressbar()
            prog.start(n_trials)
            if hasattr(prog, 'description'):
                prog.description = 'Performance testing {} (0 / {})...'.format(
                    name, n_trials
                )

        # Make sure that everything we do catches NaNs as exceptions,
        # such that we can correctly record them as failures.
        with numpy_err_policy(divide='raise'):
            # Loop through once to dispatch tasks.
            # We'll loop through again to collect results.
            results = [apply(trial_fn) for idx in range(n_trials)]

            for idx, result in enumerate(results):
                # FIXME: This is bad practice, but I don't feel like rewriting to
                #        avoid right now.
                try:
                    performance[idx, :] = result.get()
                    if prog is not None:
                        prog.update(idx)
                        if hasattr(prog, 'description'):
                            prog.description = 'Performance testing {} ({} / {})...'.format(
                                name, idx, n_trials
                            )

                except:
                    if allow_failures:
                        performance.mask[idx, :] = True
                    else:
                        raise

    finally:
        if prog is not None:
            prog.finished()

    return performance
