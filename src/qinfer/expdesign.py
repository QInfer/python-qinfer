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

# for BCRB and BED classes
import scipy.optimize as opt
from ._lib import enum

from abc import ABCMeta, abstractmethod

from qinfer.finite_difference import *

## CLASSES #####################################################################

OptimizationAlgorithms = enum.enum("NULL", "CG", "NCG")

class Heuristic(object):
    r"""
    Defines a heuristic used for selecting new experiments without explicit
    optimization of the risk. As an example, the :math:`t_k = (9/8)^k`
    heuristic discussed by [FGC12]_ does not make explicit reference to the
    risk, and so would be appropriate as a `Heuristic` subclass.
    
    Note that the design of this abstract base class is still being decided,
    such that it is a placeholder for now.
    """
    __metaclass__ = ABCMeta
    
    def __init__(self):
        raise NotImplementedError("Not yet implemented.")
    
    @abstractmethod
    def __call__(self, *args):
        raise NotImplementedError("Not yet implemented.")

class ExperimentDesigner(object):
    """
    Designs new experiments using the current best information provided by a
    Bayesian updater.
    
    :param qinfer.smc.SMCUpdater updater: A Bayesian updater to design
        experiments for.
    :param OptimizationAlgorithms opt_algo: Algorithm to be used to perform
        local optimization.
    """

    def __init__(self, updater, opt_algo=OptimizationAlgorithms.CG):
        if opt_algo not in OptimizationAlgorithms.reverse_mapping:
            raise ValueError("Unsupported or unknown optimization algorithm.")
    
        self._updater = updater
        self._opt_algo = opt_algo
        
        # Set everything up for the first experiment.
        self.new_exp()
        
    ## METHODS #################################################################
    def new_exp(self):
        """
        Resets this `ExperimentDesigner` instance and prepares for designing
        the next experiment.
        """
        self.__best_cost = None
        self.__best_ep = None
        
    def design_expparams_field(self,
            guess, field,
            cost_scale_k=1.0, disp=False, maxiter=None, store_guess=False,
            grad_h=1e-10
        ):
        r"""
        Designs a new experiment by varying a single field of a shape ``(1,)``
        record array and minimizing the objective function
        
        .. math::
            O(\vec{e}) = r(\vec{e}) + k \$(\vec{e}),
        
        where :math:`r` is the Bayes risk as calculated by the updater, and
        where :math:`\$` is the cost function specified by the model. Here,
        :math:`k` is a parameter specified to relate the units of the risk and
        the cost. See :ref:`expdesign` for more details.
        
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
        :param int maxiter: For those optimization algorithms which support
            it, limits the number of optimization iterations used for each
            guess.
        :param bool store_guess: If ``True``, will compare the outcome of this
            guess to previous guesses and then either store the optimization of
            this experiment, or the previous best-known experiment design.
        :param float grad_h: Step size to use in estimating gradients. Used
            only if ``opt_algo`` is NCG.
        :return: An array representing the best experiment design found so
            far for the current experiment.
        """
        
        # Define some short names for commonly used properties.
        up = self._updater
        m  = up.model
        
        # Generate a new guess or use a guess provided, depending on the
        # type of the guess argument.
        if isinstance(guess, Heuristic):
            raise NotImplementedError("Not yet implemented.")
        elif callable(guess):
            # Generate a new guess by calling the guess function provided.
            ep = guess(
                idx_exp=len(up.data_record),
                mean=up.est_mean(),
                cov=up.est_covariance_mtx()
            )
        else:
            # Make a copy of the guess that we can manipulate, but otherwise
            # use it as-is.
            ep = np.copy(guess)
        
        # Define an objective function that wraps a vector of scalars into
        # an appropriate record array.
        def objective_function(x):
            """
            Used internally by design_expparams_field.
            If you see this, something probably went wrong.
            """
            ep[field] = x
            return up.bayes_risk(ep) + cost_scale_k * m.experiment_cost(ep)
            
        # Some optimizers require gradients of the objective function.
        # Here, we create a FiniteDifference object to compute that for
        # us.
        d_dx_objective = FiniteDifference(objective_function, len(ep[field]))
        
        # Allocate a variable to hold the local optimum value found.
        # This way, if an optimization algorithm doesn't support returning
        # the value as well as the location, we can find it manually.
        f_opt = None
            
        # Here's the core, where we break out and call the various optimization
        # routines provided by SciPy.
        if self._opt_algo == OptimizationAlgorithms.NULL:
            # This optimization algorithm does nothing locally, but only
            # exists to leverage the store_guess functionality below.
            x_opt = guess[0][field]
            
        elif self._opt_algo == OptimizationAlgorithms.CG:
            # Prepare any additional options.
            opt_options = {}
            if maxiter is not None:
                opt_options['maxiter'] = maxiter
                
            # Actually call fmin_cg, gathering all outputs we can.
            x_opt, f_opt, func_calls, grad_calls, warnflag = opt.fmin_cg(
                objective_function, guess[0][field],
                disp=disp, full_output=True, **opt_options
            )
            
        elif self._opt_algo == OptimizationAlgorithms.NCG:
            # Prepare any additional options.
            opt_options = {}
            if maxiter is not None:
                opt_options['maxiter'] = maxiter
                
            # Actually call fmin_cg, gathering all outputs we can.
            x_opt, f_opt, func_calls, grad_calls, h_calls, warnflag = opt.fmin_ncg(
                objective_function, guess[0][field],
                d_dx_objective,
                disp=disp, full_output=True, **opt_options
            )
            
        # Optionally compare the result to previous guesses.            
        if store_guess:
            # Possibly compute the objective function value at the local optimum
            # if we don't already know it.
            if f_opt is None:
                guess_qual = objective_function(x_opt)
            
            # Compare to the known best cost so far.
            if self.__best_cost is None or (self.__best_cost > f_opt):
                # No known best yet, or we're better than the previous best,
                # so record this guess.
                ep[field] = x_opt
                self.__best_cost = f_opt
                self.__best_ep = ep
            else:
                ep = self.__best_ep # Guess is bad, return current best guess
        else:
            # We aren't using guess recording, so just pack the local optima
            # into ep for returning.
            ep[field] = x_opt
        
        # In any case, return the optimized guess.
        return ep
        
    
