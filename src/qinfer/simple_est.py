#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# simple_est.py: Simplified estimation functions for common experiments.
##
# © 2016 Chris Ferrie (csferrie@gmail.com) and
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

## EXPORTS ###################################################################

__all__ = [
    'simple_est_prec',
    'simple_est_rb',
    # TODO:
    # 'simple_est_rabi'
]

## IMPORTS ###################################################################

import numpy as np

from qinfer.smc import SMCUpdater
from qinfer.test_models import SimplePrecessionModel
from qinfer.rb import RandomizedBenchmarkingModel
from qinfer.derived_models import BinomialModel
from qinfer.distributions import UniformDistribution, PostselectedDistribution
from qinfer.resamplers import LiuWestResampler

# We want to be able to support Pandas without requiring it, so a conditional
# import is required.
try:
    import pandas as pd
except:
    pd = None

## FUNCTIONS #################################################################

def data_to_params(data,
        expparams_dtype,
        col_outcomes=(0, 'counts'),
        cols_expparams=None
    ):
    """
    Given data as a NumPy array, separates out each column either as
    the outcomes, or as a field of an expparams array. Columns may be specified
    either as indices into a two-axis scalar array, or as field names for a one-axis
    record array.

    Since scalar arrays are homogenous in type, this may result in loss of precision
    due to casting between data types.
    """
    BY_IDX, BY_NAME = range(2)

    is_exp_scalar = np.issctype(expparams_dtype)
    is_data_scalar = np.issctype(data.dtype) and not data.dtype.fields

    s_ = (
        (lambda idx: np.s_[..., idx[BY_IDX]])
        if is_data_scalar else
        (lambda idx: np.s_[idx[BY_NAME]])
    )

    outcomes = data[s_(col_outcomes)].astype(int)

    # mk new slicer t

    expparams = np.empty(outcomes.shape, dtype=expparams_dtype)
    if is_exp_scalar:
        expparams[:] = data[s_(cols_expparams)]
    else:
        for expparams_key, column in cols_expparams.items():
            expparams[expparams_key] = data[s_(column)]

    return outcomes, expparams

def load_data_or_txt(data, dtype):
    if isinstance(data, np.ndarray):
        return data
    elif pd is not None and isinstance(data, pd.DataFrame):
        return data.to_records(index=False)

    elif hasattr(data, 'read') or isinstance(data, str):
        data = np.loadtxt(data, dtype=dtype, delimiter=',')
        return data
    else:
        raise TypeError("Expected a filename, an array or a file-like object.")


def do_update(model, n_particles, prior, outcomes, expparams, return_all, resampler=None):
    updater = SMCUpdater(model, n_particles, prior,
        resampler=resampler
    )
    updater.batch_update(outcomes, expparams, resample_interval=1)

    mean = updater.est_mean()
    cov = updater.est_covariance_mtx()

    if model.n_modelparams == 1:
        mean = mean[0]
        cov = cov[0, 0]

    if not return_all:
        return mean, cov
    else:
        return mean, cov, {
            'updater': updater
        }

def simple_est_prec(data, freq_min=0.0, freq_max=1.0, n_particles=6000, return_all=False):
    """
    Estimates a simple precession (cos²) from experimental data.
    Note that this model is mainly for testing purposes, as it does not
    consider the phase or amplitude of precession, leaving only the frequency.

    :param data: Data to be used in estimating the precession frequency.
    :type data: see :ref:`simple_est_data_arg`
    :param float freq_min: The minimum feasible frequency to consider.
    :param float freq_max: The maximum feasible frequency to consider.
    :param int n_particles: The number of particles to be used in estimating
        the precession frequency.
    :param bool return_all: Controls whether additional return
        values are provided, such as the updater.

    :column counts (int): How many counts were observed at the sampled
        time.
    :column t (float): The evolutions time at which the samples
        were collected.
    :column n_shots (int): How many samples were collected at the
        given evolution time.

    :return mean: Bayesian mean estimator for the precession frequency.
    :return var: Variance of the final posterior over frequency.
    :return extra: See :ref:`simple_est_extra_return`. Only returned
        if ``return_all`` is `True`.
    """
    model = BinomialModel(SimplePrecessionModel(freq_min))
    prior = UniformDistribution([0, freq_max])

    data = load_data_or_txt(data, [
        ('counts', 'uint'),
        ('t', float),
        ('n_shots', 'uint')
    ])

    outcomes, expparams = data_to_params(data,
        model.expparams_dtype,
        cols_expparams={
            'x': (1, 't'),
            'n_meas': (2, 'n_shots')
        }
    )

    return do_update(
        model, n_particles, prior, outcomes, expparams,
        return_all
    )


def simple_est_rb(data, interleaved=False, p_min=0.0, p_max=1.0, n_particles=8000, return_all=False):
    r"""
    Estimates the fidelity of a gateset from a standard or interleaved randomized benchmarking
    experiment.
    
    :param data: Data to be used in estimating the gateset fidelity.
    :type data: see :ref:`simple_est_data_arg`
    :param float p_min: Minimum value of the parameter :math:`p`
        to consider feasible.
    :param float p_max: Minimum value of the parameter :math:`p`
        to consider feasible.
    :param int n_particles: The number of particles to be used in estimating
        the randomized benchmarking model.
    :param bool return_all: Controls whether additional return
        values are provided, such as the updater.

    :column counts (int): How many sequences of length :math:`m` were observed to
        survive.
    :column m (int): How many gates were used for sequences in this row of the data.
    :column n_shots (int): How many different sequences of length :math:`m`
        were measured.
    :column reference (bool): `True` if this row represents reference sequences, or
        `False` if the gate of interest is interleaved. Note that this column is omitted
        if ``interleaved`` is `False`.

    :return mean: Bayesian mean estimator for the model vector
        :math:`(p, A, B)`, or :math:`(\tilde{p}, p_{\text{ref}}, A, B)`
        for the interleaved case.
    :return var: Variance of the final posterior over RB model vectors.
    :return extra: See :ref:`simple_est_extra_return`. Only returned
        if ``return_all`` is `True`.
    """
    model = BinomialModel(RandomizedBenchmarkingModel(interleaved=interleaved))
    prior = PostselectedDistribution(UniformDistribution([
            [p_min, p_max],
            [0, 1],
            [0, 1]
        ] if not interleaved else [
            [p_min, p_max],
            [p_min, p_max],
            [0, 1],
            [0, 1]
        ]),
        model
    )

    data = load_data_or_txt(data, [
        ('counts', 'uint'),
        ('m', 'uint'),
        ('n_shots', 'uint')
    ] + ([
        ('reference', 'uint')
    ] if interleaved else []))

    cols_expparams = {
        'm': (1, 'm'),
        'n_meas': (2, 'n_shots')
    }
    if interleaved:
        cols_expparams['reference'] = (3, 'reference')

    outcomes, expparams = data_to_params(data,
        model.expparams_dtype,
        cols_expparams=cols_expparams
    )

    return do_update(
        model, n_particles, prior, outcomes, expparams,
        return_all
    )
