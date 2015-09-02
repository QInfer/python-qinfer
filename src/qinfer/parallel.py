#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# parallel.py: Tools for distributing computation.
##
# © 2014 Chris Ferrie (csferrie@gmail.com) and
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

## FEATURES ##################################################################

from __future__ import division # Ensures that a/b is always a float.

## EXPORTS ###################################################################

__all__ = ['DirectViewParallelizedModel']

## IMPORTS ###################################################################

import numpy as np
from qinfer.abstract_model import Model
from qinfer.derived_models import DerivedModel

import warnings

try:
    import ipyparallel as ipp
    interactive = ipp.interactive
except ImportError:
    try:
        import IPython.parallel as ipp
        interactive = ipp.interactive
    except ImportError, KeyError:
        import warnings
        warnings.warn(
            "Could not import IPython parallel. "
            "Parallelization support will be disabled."
        )
        ipp = None
        interactive = lambda fn: fn

## LOGGING ###################################################################

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
    
## CLASSES ###################################################################

class DirectViewParallelizedModel(DerivedModel):
    r"""
    Given an instance of a `Model`, parallelizes execution of that model's
    likelihood by breaking the ``modelparams`` array into segments and
    executing a segment on each member of a :ref:`~IPython.parallel.DirectView`.
    
    This :ref:`Model` assumes that it has ownership over the DirectView, such
    that no other processes will send tasks during the lifetime of the Model.
    
    TODO: describe parameters.

    :param bool purge_client: If ``True``, then this model will purge results
        and metadata from the IPython client whenever the model cache is cleared.
        This is useful for solving memory leaks caused by very large numbers of
        calls to ``likelihood``. By default, this is disabled, since enabling
        this option can cause data loss if the client is being sent other tasks
        during the operation of this model.
    """
    
    ## INITIALIZER ##
    
    def __init__(self, serial_model, direct_view, purge_client=False):
        if ipp is None:
            raise RuntimeError(
                "This model requires IPython parallelization support, "
                "but an error was raised importing IPython.parallel."
            )

        self._dv = direct_view
        self._purge_client = purge_client
        
        super(DirectViewParallelizedModel, self).__init__(serial_model)
    
    ## SPECIAL METHODS ##
    
    def __getstate__(self):
        # Since instances of this class will be pickled as they are passed to
        # remote engines, we need to be careful not to include _dv
        return {
            '_underlying_model': self._underlying_model,
            '_dv': None,
            '_call_count': self._call_count,
            '_sim_count': self._sim_count
        }
    
    ## PROPERTIES ##

    # Provide _serial_model as a back-compat.
    @property
    def _serial_model(self):
        warnings.warn("_serial_model is deprecated in favor of _underlying_model.",
            DeprecationWarning
        )
        return self._underlying_model
    @_serial_model.setter
    def _serial_model(self, value):
        warnings.warn("_serial_model is deprecated in favor of _underlying_model.",
            DeprecationWarning
        )
        self._underlying_model = value
    

    @property
    def n_engines(self):
        return len(self._dv) if self._dv is not None else 0
            
    ## METHODS ##
    
    def clear_cache(self):
        self.underlying_model.clear_cache()
        try:
            logger.info('DirectView results has {} items. Clearing.'.format(
                len(self._dv.results)
            ))
            self._dv.purge_results('all')
            if self._purge_client:
                self._dv.client.purge_everything()
        except:
            pass
    
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(DirectViewParallelizedModel, self).likelihood(outcomes, modelparams, expparams) 
        
        if self._dv is None:
            raise RuntimeError(
                "No direct view provided; this may be because the instance was "
                "loaded from a pickle or NumPy saved array without providing a "
                "new direct view."
            )

        # Need to decorate with interactive to overcome namespace issues with
        # remote engines.
        @interactive
        def serial_likelihood(mps, sm, os, eps):
            return sm.likelihood(os, mps, eps)

        # TODO: check whether there's a better way to pass the extra parameters
        # that doesn't use so much memory.
        # The trick is that serial_likelihood will be pickled, so we need to be
        # careful about closures.
        L = self._dv.map_sync(
            serial_likelihood,
            np.array_split(modelparams, self.n_engines, axis=0),
            [self.underlying_model] * self.n_engines,
            [outcomes] * self.n_engines,
            [expparams] * self.n_engines,
        )

        return np.concatenate(L, axis=1)

