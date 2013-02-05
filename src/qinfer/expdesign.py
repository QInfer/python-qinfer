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
    'Heuristic',
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

class Heuristic(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Not yet implemented.")

class ExperimentDesigner(object):
    # TODO: docstring!

    def __init__(self, updater, opt_algo=OptimizationAlgorithms.CG):
        self._updater = updater
        self._opt_algo = opt_algo # TODO: check that opt_algo is valid.
	self.__best_cost=None
        
    ## METHODS #################################################################
    def new_exp(self):
	self.__best_cost=None
	
        
    def design_expparams_field(self, guess, field, cost_scale_k=1.0, disp=False, maxiter=None,store_guess=False):
        """
        TODO
        
        :param guess: Either a record array with a single guess, or
            a callable function that generates guesses.
        :type guess: Instance of :class:`~Heuristic`, `callable`
            or :class:`~numpy.ndarray` of ``dtype``
            :attr:`~qinfer.abstract_model.Simulatable.expparams_dtype`
        :param str field: The name of the ``expparams`` field to be optimized.
            All other fields of ``guess`` will be held constant.
        :param float cost_scale_k: A scale parameter :math:`k` relating the
            Bayes risk to the experiment cost.
            See :ref:`expdesign`.
        :param bool disp: If `True`, the optimization will print additional
            information as it proceeds.
        :return: TODO
        """
        
        # TODO: this method is a definite WIP.
        up = self._updater
        m  = up.model
        
        if isinstance(guess, Heuristic):
            raise NotImplementedError("Not yet implemented.")
        elif callable(guess):
            ep = guess(idx_exp=len(up.data_record), mean=up.est_mean(), cov=up.est_covariance_mtx())
        else:
            # Make a copy of the guess that we can manipulate.
            ep = np.copy(guess)
        
        def objective_function(x):
            ep[field] = x
            return up.bayes_risk(ep) + cost_scale_k * m.experiment_cost(ep)
            
        if self._opt_algo == OptimizationAlgorithms.CG:
            opt_options = {}
            if maxiter is not None:
                opt_options['maxiter'] = maxiter
            # TODO: form initial guesses.
            # TODO: Optimize each according to objective_function
            x_opt = opt.fmin_cg(objective_function, guess[0][field], disp=disp, **opt_options)
	    
	    if store_guess:
		guess_qual=objective_function(x_opt)
		if self.__best_cost == None :
		    ep[field] = x_opt
		    self.__best_cost=guess_qual #Stores best guess
		    self.__best_ep=ep
		elif (self.__best_cost > guess_qual):
		    ep[field]=x_opt
		    self.__best_cost=guess_qual
		    self.__best_ep=ep
		else:
		    ep=self.__best_ep # Guess is bad, return current best guess
	    else:
		ep[field]=x_opt
            
            return ep
        elif self._opt_algo == OptimizationAlgorithms.NCG:
            raise NotImplementedError("NCG optimization algorithm not yet implemented.")
        
    
