#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# expdesign.py: Adaptive experimental design algorithms.
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

## FEATURES ####################################################################

from __future__ import division

## ALL #########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
    'ExperimentDesigner'
]

## IMPORTS #####################################################################

import numpy as np
import warnings

from abstract_model import Simulatable, Model, DifferentiableModel

# for BCRB and BED classes
import scipy.linalg as la
import scipy.optimize as opt
from utils import outer_product, particle_meanfn, particle_covariance_mtx, mvee, uniquify

## CLASSES #####################################################################

class ExperimentDesigner(object):
    # TODO: docstring!
    
    def __init__(self, updater):
        self._updater = updater
        
    ## METHODS #################################################################
        
    def design_expparams_field(self, field, other_fields=None, cost_scale_k=1.0):
        # TODO: this method is a definite WIP.
        up = self._updater
        m  = up.model
        
        def objective_function(x):
            ep = np.empty((1,), dtype=m.expparams_dtype)
            # TODO: set fields of ep based on other_fields.
            ep[field] = x
            return up.bayes_risk(ep) + cost_scale_k * m.experiment_cost(ep)
            
        # TODO: feed the objective_function above into the appropriate
        #       optimizer.
        
        
    
