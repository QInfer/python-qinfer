#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# noisycoin_example.py: noisy coin model performance testing module
##
# © 2013 Chris Ferrie (csferrie@gmail.com) and
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

"""
Usage: noisycoin.py <n-particles> <n-exp> [options]

Arguments:
    n_particles    Number of SMC particles to use.
    n_exp          Number of experiments to perform.
    
Options:
    -h             Show this screen and exit.
    -o FILE        Save performance data to a file.
    --plot         Plots the results.
    --smc          Enable SMC performance measurements.
    --smcale       Enable SMC-ALE performance measurements.
    --n-sim=N      Number of simulated runs to average over. [default: 100]
    --err_tol=e    Error tolerance for ALE [default: 0.1]
    --hedge=h      Hedging parameter for ALE [default: 0.5]
    --alpha=A      Value of alpha to assume during experiments. [default: 0.85]
    --beta=B       Value of beta to assume during experiments. [default: 0.05]
"""

## FEATURES ####################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

## IMPORTS #####################################################################

from builtins import range

import numpy as np
import time
from scipy.special import gammaln, betaln
import sys

## Imports from within QInfer. ##
from .. import distributions, ale
from ..smc import SMCUpdater
from ..test_models import NoisyCoinModel
from .._lib import docopt

try:
    from .. import dialogs
except ImportError:
    print("[WARN] Could not import dialogs.")
    dialogs = None

## Exact ######################################################################

def exactBME(k, K, a, b, gamma=1):
    idx_k = np.arange(k+1)    
    idx_K = np.arange(K-k+1)[np.newaxis].transpose()
    
    numerator   = (
        gammaln(k+1) - gammaln(idx_k+1) - gammaln(k-idx_k+1) + gammaln(K-k+1) -
        gammaln(idx_K+1) - gammaln(K-k-idx_K+1) + (idx_k+idx_K)*np.log(a-b) +
        (k-idx_k)*np.log(b) + (K-k-idx_K)*np.log(1-a) +
        betaln(idx_k+gamma+1,idx_K+gamma)
    )
    denominator = (
        gammaln(k+1) - gammaln(idx_k+1) - gammaln(k-idx_k+1) + gammaln(K-k+1) -
        gammaln(idx_K+1) - gammaln(K-k-idx_K+1) + (idx_k+idx_K)*np.log(a-b) +
        (k-idx_k)*np.log(b) + (K-k-idx_K)*np.log(1-a) +
        betaln(idx_k+gamma,idx_K+gamma)
    )
    bme = np.sum(np.exp(numerator))/np.sum(np.exp(denominator))
        
    var = np.sum(np.exp(
            numerator - betaln(idx_k+gamma+1,idx_K+gamma) +
            betaln(idx_k + gamma + 2, idx_K + gamma)
        )) / np.sum(np.exp(denominator)) - bme ** 2
    
    return bme, var

## SCRIPT ######################################################################

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Handle and unpack command-line arguments.
    args = docopt.docopt(__doc__, sys.argv[1:])
    
    save_fname  = args["-o"]
    
    n_sim       = int(args["--n-sim"])
    n_particles = int(args["<n-particles>"])
    n_exp       = int(args["<n-exp>"])
    
    alpha       = float(args["--alpha"])
    beta        = float(args["--beta"])
    err_tol     = float(args["--err_tol"])    
    hedge       = float(args["--hedge"])    
    
    do_smc      = bool(args["--smc"])
    do_ale      = bool(args["--smcale"])
    
    do_plot     = bool(args["--plot"])
    
    if not (do_smc or do_ale):
        raise ValueError("At least one of SMC or SMC-ALE must be enabled.")
            
    # Model and prior initialization
    prior = distributions.UniformDistribution([0, 1])
    model = NoisyCoinModel()

    # Make a dict of updater constructors. This will define what kinds
    # of perfomance data we care about.
    updater_ctors = dict()
    if do_smc:
        updater_ctors['SMC'] = lambda: SMCUpdater(
            model, n_particles, prior
        )
    if do_ale:
        ale_model = ale.ALEApproximateModel(model,
            error_tol=err_tol,
            est_hedge=hedge,
            adapt_hedge=hedge
        )
        updater_ctors['SMC_ALE'] = lambda: SMCUpdater(
            ale_model, n_particles, prior
        )

    # Store alpha and beta into the expparams array where the will
    # remain for the rest of the program.
    expparams = np.array([(alpha, beta)], dtype=model.expparams_dtype)
    
    # Make a dtype for holding performance data in a record array.
    # Note that we could do this with out record arrays, but it's easy to
    # use field names this way.
    performance_dtype = [
        ('est_mean', 'f8'), ('est_cov_mat', 'f8'),
        ('true_err', 'f8'), ('resample_count', 'i8'),
        ('elapsed_time', 'f8'),
        ('like_count', 'i8'), ('sim_count', 'i8'),
    ]
    
    # Create arrays to hold the data that we obtain and the true models.
    true_param = np.zeros(n_sim)
    outcomes = np.zeros([n_sim, n_exp], dtype=int)
    
    # Create arrays to hold performance histories and store them
    # in a dict.
    performance_hist = {
        updater_name: np.zeros((n_sim, n_exp), dtype=performance_dtype)
        for updater_name in updater_ctors
    }

    # Create arrays to compare to the ideal known-best BMEs.
    bme = np.zeros([n_sim, n_exp])
    var = np.zeros([n_sim, n_exp])
    bme_err = np.zeros([n_sim, n_exp])
     
    # Possibly prepare a dialog.
    if dialogs is not None:
        progress = dialogs.ProgressDialog(
            task_title="Noisy Coin Demo",
            task_status="0 / {n_sim} runs completed.".format(n_sim=n_sim),
            maxprog=n_sim)
    else:
        progress = None
     
    # Now we run the Monte Carlo simulations.
    for idx_sim in range(n_sim):
        
        # First, make new updaters using the constructors
        # defined above.
        updaters = {
            updater_name: updater_ctors[updater_name]()
            for updater_name in updater_ctors
        }        
        
        # Sample true set of modelparams.
        truemp = prior.sample()
        true_param[idx_sim] = truemp

        # Now loop over experiments, updating each of the
        # updaters with the same data, so that we can compare
        # their estimation performance.
        for idx_exp in range(n_exp):
            
            # Make a short hand for indexing the current simulation
            # and experiment.
            idxs = np.s_[idx_sim, idx_exp]
            
            # Start by simulating and recording the data.
            outcome = model.simulate_experiment(truemp, expparams)
            outcomes[idxs] = outcome
            
            # Next, feed this data into each updater in turn.
            for name, updater in updaters.iteritems():
                # Reset the like_count and sim_count
                # properties so that we can count how many were used
                # by this update. Note that this is a hack;
                # an appropriate method should be added to
                # Simulatable.
                model._sim_count = 0
                model._call_count = 0
                
                # Time the actual update.
                tic = toc = None
                tic = time.time()
                updater.update(outcome, expparams)
                performance_hist[name]['elapsed_time'][idxs] = \
                    time.time() - tic
            
                # Record the performance of this updater.
                est_mean = updater.est_mean()
                performance_hist[name][idxs]['est_mean'] = \
                    est_mean
                performance_hist[name][idxs]['true_err'] = \
                    np.abs(est_mean - truemp) ** 2
                performance_hist[name][idxs]['est_cov_mat'] = \
                    updater.est_covariance_mtx()
                performance_hist[name][idxs]['resample_count'] = \
                    updater.resample_count
                performance_hist[name][idxs]['like_count'] = \
                    model.call_count
                performance_hist[name][idxs]['sim_count'] = \
                    model.sim_count
            
            # Finally, record the ideal stats.
            bme[idxs], var[idxs] = exactBME(
                idx_exp + 1 - np.sum(outcomes[idx_sim]), idx_exp + 1,
                alpha, beta
            )
            bme_err[idxs] = np.abs(bme[idx_sim, idx_exp] - truemp) ** 2
            
        # Notify the user of any progress.
        if progress is not None:
            progress.progress = idx_sim + 1
            progress.status = "{} / {n_sim} runs completed.".format(
                idx_sim + 1, n_sim=n_sim
            )
            
    # Now that we're done, delete the progress bar.
    if progress is not None:
        progress.complete()
        progress.close()
    
    if save_fname is not None:
        np.savez(save_fname, **performance_hist)
            
    # Now, if we've been asked to plot, go on and make a figure.
    if do_plot:
    
        # We'll only need the one figure window.
        fig = plt.figure()
        
        # Calculate the average error over all trials for each updater.
        avg_error = {
            name: np.average(hist['true_err'], 0)
            for name, hist in performance_hist.iteritems()
        }
        # Also compare to the average error for the Bayesian mean estimator.
        avg_error['BME'] = np.average(bme_err, 0)

        # Plot that against the asymptotic optimum.
        arr_exp = np.arange(n_exp) + 1
        asympt = (1/6) / (alpha-beta)**2 / arr_exp - (1/6) / (alpha-beta)**2 / arr_exp**2
        
        # Now actually plot everything on a loglog scale.
        if do_smc:
            plt.loglog(avg_error['SMC'], c='blue', label='SMC')
        if do_ale:
            plt.loglog(avg_error['SMC_ALE'], c='green', label='LFQIA')
            
        plt.loglog(avg_error['BME'], c='red', label='Exact')
        plt.loglog(asympt, c='black', label='Asymptotic')
        plt.legend()

        # Finally, show the plot.
        plt.show()
    
