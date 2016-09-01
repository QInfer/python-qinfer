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

## FEATURES ###################################################################

from __future__ import absolute_import
from __future__ import division

## ALL ########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
    'ExperimentDesigner',
    'Heuristic',
    'EnsembleHeuristic',
    'ExpSparseHeuristic',
    'PGH',
    'OptimizationAlgorithms'
]

## IMPORTS ####################################################################

from future.utils import with_metaclass

import numpy as np

# for BCRB and BED classes
import scipy.optimize as opt
from qinfer._lib import enum # <- TODO: replace with flufl.enum!

from abc import ABCMeta, abstractmethod
import warnings

from qinfer.finite_difference import *

## FUNCTIONS ###################################################################

def identity(arg): return arg

## CLASSES #####################################################################

OptimizationAlgorithms = enum.enum("NULL", "CG", "NCG", "NELDER_MEAD")

class Heuristic(with_metaclass(ABCMeta, object)):
    r"""
    Defines a heuristic used for selecting new experiments without explicit
    optimization of the risk. As an example, the :math:`t_k = (9/8)^k`
    heuristic discussed by [FGC12]_ does not make explicit reference to the
    risk, and so would be appropriate as a `Heuristic` subclass.
    In particular, the [FGC12]_ heuristic is implemented by the
    :class:`ExpSparseHeuristic` class.
    """

    def __init__(self, updater):
        self._updater = updater
    
    @abstractmethod
    def __call__(self, *args):
        raise NotImplementedError("Not yet implemented.")

class EnsembleHeuristic(Heuristic):
    r"""
    Heuristic that randomly chooses one of several other
    heuristics.

    :param list ensemble: List of tuples ``(heuristic, pr)``
        specifying the probability of choosing each member
        heuristic.
    """

    def __init__(self, ensemble):
        self._pr = np.array([pr for heuristic, pr in ensemble])
        self._heuristics = ([heuristic for heuristic, pr in ensemble])

    def __call__(self, *args):
        idx_heuristic = np.random.choice(len(self._heuristics), p=self._pr)
        return self._heuristics[idx_heuristic](*args)
        
class ExpSparseHeuristic(Heuristic):
    r"""
    Implements the exponentially-sparse time evolution heuristic
    of [FGC12]_, under which :math:`t_k = A b^k`, where :math:`A`
    and :math:`b` are parameters of the heuristic.

    :param qinfer.smc.SMCUpdater updater: Posterior updater for which
        experiments should be heuristicly designed.
    :param float scale: The value of :math:`A`, implicitly setting
        the frequency scale for the problem.
    :param float base: The base of the exponent; in general, should
        be closer to 1 for higher-dimensional models.
    :param str t_field: Name of the expparams field representing time.
        If None, then the generated expparams are taken to be scalar,
        and not a record.
    :param dict other_fields: Values of the other fields to be used
        in designed experiments.
    """

    def __init__(self,
            updater, scale=1, base=9/8,
            t_field=None, other_fields=None
        ):
        super(ExpSparseHeuristic, self).__init__(updater)
        self._scale = scale
        self._base = base
        self._t_field = t_field
        self._other_fields = other_fields

    def __call__(self):
        n_exps = len(self._updater.data_record)
        t = self._scale * (self._base ** n_exps)
        dtype = self._updater.model.expparams_dtype

        if self._t_field is None:
            return np.array([t], dtype=dtype)
        else:
            eps = np.empty((1,), dtype=dtype)
            for field, value in self._other_fields.items():
                eps[field] = value
            eps[self._t_field] = t
            return eps

class PGH(Heuristic):
    """
    Implements the *particle guess heuristic* (PGH) of [WGFC13a]_, which
    selects two particles from the current posterior, selects one as an
    inversion hypothesis and sets the time parameter to be the inverse of
    the distance between the particles. In this way, the PGH adapts to the
    current uncertianty without additional simulation resources.
    
    :param qinfer.smc.SMCUpdater updater: Posterior updater for which
        experiments should be heuristicly designed.
    :param str inv_field: Name of the ``expparams`` field corresponding to the
        inversion hypothesis.
    :param str t_field: Name of the ``expparams`` field corresponding to the
        evolution time.
    :param callable inv_func: Function to be applied to modelparameter vectors
        to produce an inversion field ``x_``.
    :param callable t_func: Function to be applied to the evolution time to produce a
         time field ``t``.
    :param int maxiters: Number of times to try and choose distinct particles
        before giving up.
    :param dict other_fields: Values to set for fields not given by the PGH.
    
    Once initialized, a ``PGH`` object can be called to generate a new
    experiment parameter vector:
    
    >>> pgh = PGH(updater) # doctest: +SKIP
    >>> expparams = pgh() # doctest: +SKIP
    
    If the posterior weights are very highly peaked (that is, if the effective
    sample size is too small, as measured by
    :attr:`~qinfer.smc.SMCUpdater.n_ess`), then it may be the case that the two
    particles chosen by the PGH are identical, such that the time would be
    determined by ``1 / 0``. In this case, the `PGH` class will instead draw
    new pairs of particles until they are not identical, up to ``maxiters``
    attempts. If that limit is reached, a `RuntimeError` will be raised.
    """
    
    def __init__(self, updater, inv_field='x_', t_field='t',
                 inv_func=identity,
                 t_func=identity,
                 maxiters=10,
                 other_fields=None
                 ):
        super(PGH, self).__init__(updater)
        self._x_ = inv_field
        self._t = t_field
        self._inv_func = inv_func
        self._t_func = t_func
        self._maxiters = maxiters
        self._other_fields = other_fields if other_fields is not None else {}
        
    def __call__(self):
        idx_iter = 0
        while idx_iter < self._maxiters:
                
            x, xp = self._updater.sample(n=2)[:, np.newaxis, :]
            if self._updater.model.distance(x, xp) > 0:
                break
            else:
                idx_iter += 1
                
        if self._updater.model.distance(x, xp) == 0:
            raise RuntimeError("PGH did not find distinct particles in {} iterations.".format(self._maxiters))
            
        eps = np.empty((1,), dtype=self._updater.model.expparams_dtype)
        eps[self._x_] = self._inv_func(x)
        eps[self._t]  = self._t_func(1 / self._updater.model.distance(x, xp))
        
        for field, value in self._other_fields.items():
            eps[field] = value
        
        return eps

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
        
    ## METHODS ################################################################
    def new_exp(self):
        """
        Resets this `ExperimentDesigner` instance and prepares for designing
        the next experiment.
        """
        self.__best_cost = None
        self.__best_ep = None
        
    def design_expparams_field(self,
            guess, field,
            cost_scale_k=1.0, disp=False,
            maxiter=None, maxfun=None,
            store_guess=False, grad_h=None, cost_mult=False
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
            it (currently, only CG and NELDER_MEAD), limits the number of
            optimization iterations used for each guess.
        :param int maxfun: For those optimization algorithms which support it
            (currently, only NCG and NELDER_MEAD), limits the number of
            objective calls that can be made.
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
        if (cost_mult==False):
            def objective_function(x):
                """
                Used internally by design_expparams_field.
                If you see this, something probably went wrong.
                """
                ep[field] = x
                return up.bayes_risk(ep) + cost_scale_k * m.experiment_cost(ep)
        else:
            def objective_function(x):
                """
                Used internally by design_expparams_field.
                If you see this, something probably went wrong.
                """
                ep[field] = x
                return up.bayes_risk(ep)* m.experiment_cost(ep)**cost_scale_k
        
            
        # Some optimizers require gradients of the objective function.
        # Here, we create a FiniteDifference object to compute that for
        # us.
        d_dx_objective = FiniteDifference(objective_function, ep[field].size)
        
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
            if maxfun is not None:
                opt_options['maxfun'] = maxfun
            if grad_h is not None:
                opt_options['epsilon'] = grad_h
                
            # Actually call fmin_tnc, gathering all outputs we can.
            # We use fmin_tnc in preference to fmin_ncg, as they implement the
            # same algorithm, but fmin_tnc seems better behaved with respect
            # to very flat gradient regions, due to respecting maxfun.
            # By contrast, fmin_ncg can get stuck in an infinite loop in
            # versions of SciPy < 0.11.
            #
            # Note that in some versions of SciPy, there was a bug in
            # fmin_ncg and fmin_tnc that can propagate outward if the gradient
            # is too flat. We catch it here and return the initial guess in that
            # case, since by hypothesis, it's too flat to make much difference
            # anyway.
            try:
                x_opt, f_opt, func_calls, grad_calls, h_calls, warnflag = opt.fmin_tnc(
                    objective_function, guess[0][field],
                    fprime=None, bounds=None, approx_grad=True,
                    disp=disp, full_output=True, **opt_options
                )
            except TypeError:
                warnings.warn(
                    "Gradient function too flat for NCG.",
                    RuntimeWarning)
                x_opt = guess[0][field]
                f_opt = None
                
        elif self._opt_algo == OptimizationAlgorithms.NELDER_MEAD:
            opt_options = {}
            if maxfun is not None:
                opt_options['maxfun'] = maxfun
            if maxiter is not None:
                opt_options['maxiter'] = maxiter
                
            x_opt, f_opt, iters, func_calls, warnflag = opt.fmin(
                objective_function, guess[0][field],
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
        
    
