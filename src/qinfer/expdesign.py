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
    'ExperimentDesigner',
    'OptimizationAlgorithms'
]

## IMPORTS #####################################################################

import numpy as np
import warnings

from abstract_model import Simulatable, Model, DifferentiableModel

# for BCRB and BED classes
import scipy.linalg as la
import scipy.optimize as opt
from utils import outer_product, particle_meanfn, particle_covariance_mtx, mvee, uniquify
from ._lib import enum

## CLASSES #####################################################################

OptimizationAlgorithms = enum.enum("CG", "NCG")

class ExperimentDesigner(object):
    # TODO: docstring!

    def __init__(self, updater, opt_algo=OptimizationAlgorithms.CG):
        self._updater = updater
        self._opt_algo = opt_algo # TODO: check that opt_algo is valid.
        
    ## METHODS #################################################################
        
    def design_expparams_field(self, guess,field, other_fields=None, cost_scale_k=1.0):
        # TODO: this method is a definite WIP.
        up = self._updater
        m  = up.model
        
        def objective_function(x):
            ep = np.empty((1,), dtype=m.expparams_dtype)
            # TODO: set fields of ep based on other_fields.
            ep[field] = x
            return up.bayes_risk(ep) + cost_scale_k * m.experiment_cost(ep)
            
        if self._opt_algo == OptimizationAlgorithms.CG:
            # TODO: form initial guesses.
            # TODO: Optimize each according to objective_function
            #raise NotImplementedError("CG opt algo not yet implemented.")
	    return scipy.optimize.fmin_cg(objective_function,guess[0][field], maxiter=10)
        elif self._opt_algo == OptimizationAlgorithms.NCG:
            raise NotImplementedError("NCG optimization algorithm not yet implemented.")
        
    
