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

## FEATURES ####################################################################

from __future__ import division # Ensures that a/b is always a float.

## IMPORTS #####################################################################

import numpy as np
import IPython.parallel
from abstract_model import Model
    
## CLASSES #####################################################################

class DirectViewParallelizedModel(Model):
    r"""
    Given an instance of a `Model`, parallelizes execution of that model's
    likelihood by breaking the ``modelparams`` array into segments and
    executing a segment on each member of a :ref:`~IPython.parallel.DirectView`.
    
    This :ref:`Model` assumes that it has ownership over the DirectView, such
    that no other processes will send tasks during the lifetime of the Model.
    
    TODO: describe parameters.
    """
    
    ## INITIALIZER ##
    
    def __init__(self, serial_model, direct_view):
        self._serial_model = serial_model
        self._dv = direct_view
        
        super(DirectViewParallelizedModel, self).__init__()
    
    ## SPECIAL METHODS ##
    
    def __getstate__(self):
        return {
            '_serial_model': self._serial_model,
            '_dv': None
        }
    
    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return self._serial_model.n_modelparams
        
    @property
    def expparams_dtype(self):
        return self._serial_model.expparams_dtype
    
    @property
    def is_n_outcomes_constant(self):
        return self._serial_model.is_n_outcomes_constant

    @property
    def n_engines(self):
        return len(self._dv) if self._dv is not None else 0
        
    @property
    def modelparam_names(self):
        return self._serial_model.modelparam_names
    
    ## METHODS ##
    
    def are_models_valid(self, modelparams):
        return self._serial_model.are_models_valid(modelparams)
    
    def n_outcomes(self, expparams):
        return self._serial_model.n_outcomes(expparams)
    
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
        @IPython.parallel.interactive
        def serial_likelihood(mps, sm, os, eps):
            return sm.likelihood(os, mps, eps)

        # TODO: check whether there's a better way to pass the extra parameters
        # that doesn't use so much memory.
        # The trick is that serial_likelihood will be pickled, so we need to be
        # careful about closures.
        L = self._dv.map_sync(
            serial_likelihood,
            np.array_split(modelparams, self.n_engines, axis=0),
            [self._serial_model] * self.n_engines,
            [outcomes] * self.n_engines,
            [expparams] * self.n_engines,
        )
        return np.concatenate(L, axis=1)

