#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# qubit_tomography_example.py: qubit tomography performance module
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

## DOCUMENTATION ###############################################################

"""
Usage: qubit_tomography_example.py <n-particles> <n-exp> [options]

Arguments:
    n_particles    Number of SMC particles to use.
    n_exp          Number of experiments to perform.
    
Options:
    -h             Show this screen and exit.
    -o FILE        Save performance data to a file.
    --smc          Enable SMC performance measurements.
    --smcale       Enable SMC-ALE performance measurements.
    --n-sim=N      Number of simulated runs to average over. [default: 1000]
    --err_tol=e    Error tolerance for ALE [default: 0.1]
    --hedge=h      Hedging parameter for ALE [default: 0.5]
"""

## FEATURES ####################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

## IMPORTS #####################################################################

from builtins import range

import numpy as np
import time
import sys

## Imports from within QInfer. ##
from qinfer import distributions, ale
from qinfer.smc import SMCUpdater
from qinfer.tomography import QubitStatePauliModel
from qinfer._lib import docopt

try:
    from .. import dialogs
except ImportError:
    print("[WARN] Could not import dialogs.")
    dialogs = None

## SCRIPT ######################################################################

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
    # Handle and unpack command-line arguments.
    args = docopt.docopt(__doc__, sys.argv[1:])
    
    save_fname  = args["-o"]
    
    n_sim       = int(args["--n-sim"])
    n_particles = int(args["<n-particles>"])
    n_exp       = int(args["<n-exp>"])
    
    err_tol     = float(args["--err_tol"])    
    hedge       = float(args["--hedge"])    
    
    do_smc      = bool(args["--smc"])
    do_ale      = bool(args["--smcale"])
    if not (do_smc or do_ale):
        raise ValueError("At least one of SMC or SMC-ALE must be enabled.")
            
    # Model and prior initialization.
    prior = distributions.HilbertSchmidtUniform()
    model = QubitStatePauliModel()
    expparams = np.array([
        ([1, 0, 0], 1), # Records are indicated by tuples.
        ([0, 1, 0], 1),
        ([0, 0, 1], 1)
    ], dtype=model.expparams_dtype)
    
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
    true_param = np.zeros([n_sim,3])
    outcomes = np.zeros([n_sim, n_exp], dtype=int)
    
    # Create arrays to hold performance histories and store them
    # in a dict.
    performance_hist = {
        updater_name: np.zeros((n_sim, n_exp), dtype=performance_dtype)
        for updater_name in updater_ctors
    }
     
    # Possibly prepare a dialog.
    if dialogs is not None:
        progress = dialogs.ProgressDialog(
            task_title="Qubit Demo",
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
        truemp = np.array([prior.sample()])
        true_param[idx_sim,:] = truemp
        

        # Now loop over experiments, updating each of the
        # updaters with the same data, so that we can compare
        # their estimation performance.
        for idx_exp in range(n_exp):
            
            # Make a short hand for indexing the current simulation
            # and experiment.
            idxs = np.s_[idx_sim, idx_exp]
            
            # Start by simulating and recording the data.
            thisexp = expparams[np.newaxis, np.random.randint(0, 3)]
            outcome = model.simulate_experiment(truemp, thisexp)
            #TODO: record expparams as well
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
                updater.update(outcome, thisexp)
                performance_hist[name]['elapsed_time'][idxs] = \
                    time.time() - tic
            
                # Record the performance of this updater.
                est_mean = updater.est_mean()
                performance_hist[name][idxs]['est_mean'] = \
                    est_mean
                performance_hist[name][idxs]['true_err'] = \
                    np.sum(np.abs(est_mean - truemp) ** 2)
                performance_hist[name][idxs]['est_cov_mat'] = \
                    updater.est_covariance_mtx()
                performance_hist[name][idxs]['resample_count'] = \
                    updater.resample_count
                performance_hist[name][idxs]['like_count'] = \
                    model.call_count
                performance_hist[name][idxs]['sim_count'] = \
                    model.sim_count
            
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
            
    # TODO: Fix the plotting code. 
            
    fig = plt.figure()
    avg_error = {
        name: np.average(hist['true_err'], 0)
        for name, hist in performance_hist.iteritems()
    }
    
    if do_smc:
        plt.loglog(avg_error['SMC'], c='blue', label='SMC')
    if do_ale:
        plt.loglog(avg_error['SMC_ALE'], c='purple', label='SMC-ALE')
        
    plt.legend()

    plt.show()
