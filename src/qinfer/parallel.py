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
    
    ## METHODS ##
    
    def are_models_valid(self, modelparams):
        return self._serial_model.are_models_valid(modelparams)
    
    def n_outcomes(self, expparams):
        return self._serial_model.n_outcomes(expparams)
    
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(DirectViewParallelizedModel, self).likelihood(outcomes, modelparams, expparams)
        
        L = self._dv.map_sync(
            lambda mps, sm=self._serial_model, os=outcomes, eps=expparams:
            sm.likelihood(os, mps, eps),
            np.array_split(modelparams, len(self._dv), axis=0)            
        )
        return np.concatenate(L, axis=1)

