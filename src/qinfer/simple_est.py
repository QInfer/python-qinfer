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
    # TODO:
    # 'simple_est_rb',
    # 'simple_est_rabi'
]

## IMPORTS ###################################################################

import numpy as np

from qinfer.smc import SMCUpdater
from qinfer.test_models import SimplePrecessionModel
from qinfer.derived_models import BinomialModel
from qinfer.distributions import UniformDistribution
from qinfer.resamplers import LiuWestResampler

## FUNCTIONS #################################################################

def data_to_params(data, expparams_dtype, col_idx_outcomes, col_map_expparams=None, col_idx_expparams=None):
    """
    Given data as a NumPy array, separates out each column either as
    the outcomes, or as a field of an expparams array. Columns may be specified
    either as indices into a two-axis scalar array, or as field names for a one-axis
    record array.

    Since scalar arrays are homogenous in type, this may result in loss of precision
    due to casting between data types.
    """

    is_scalar_dtype = np.issctype(expparams_dtype)

    outcomes = data[..., col_idx_outcomes].astype(int)
    expparams = np.empty(outcomes.shape, dtype=expparams_dtype)

    if is_scalar_dtype:
        expparams[:] = data[..., col_idx_expparams]
    else:
        for expparams_key, col_idx in col_map_expparams.items():
            expparams[expparams_key] = data[..., col_idx]

    return outcomes, expparams

def load_data_or_txt(data, dtype):
    if isinstance(data, np.ndarray):
        return data
    elif hasattr(data, 'read') or isinstance(data, 'str'):
        data = np.loadtxt(data, dtype=dtype)
        return data
    else:
        raise TypeError("Expected a filename, an array or a file-like object.")

def simple_est_prec(data, freq_min=0.0, freq_max=1.0, n_particles=2000, return_all=False):
    """
    Estimates a simple precession (cos²) from experimental data.
    The columns of the data are assumed to be [counts, t, n_shots].
    Note that this model is mainly for testing purposes, as it does not
    consider the phase or amplitude of precession, leaving only the frequency.

    :param data:
    :type data: file, `str` or array
    """
    model = BinomialModel(SimplePrecessionModel(freq_min))
    prior = UniformDistribution([0, freq_max])

    data = load_data_or_txt(data, [
        ('counts', 'uint'),
        ('t', float),
        ('n_shots', 'uint')
    ])

    if np.issctype(data.dtype):
        # Loaded as a two-axis array.
        outcomes, expparams = data_to_params(data,
            model.expparams_dtype,
            col_idx_outcomes=0,
            col_map_expparams={
                'x': 1,
                'n_meas': 2
            }
        )
    else:
        # Loaded using column names.
        outcomes, expparams = data_to_params(data,
            model.expparams_dtype,
            col_idx_outcomes=0,
            col_map_expparams={
                'x': 't',
                'n_meas': 'n_shots'
            }
        )

    updater = SMCUpdater(model, n_particles, prior,
        # resampler=LiuWestResampler(a=0.9)
    )
    updater.batch_update(outcomes, expparams, resample_interval=1)

    mean = updater.est_mean()
    cov = updater.est_covariance_mtx()

    if not return_all:
        return mean, cov
    else:
        return mean, cov, {
            'updater': updater
        }

