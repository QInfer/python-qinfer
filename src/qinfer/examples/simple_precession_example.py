#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# simple_precession_example.py
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
Usage: simple_precession_example <n_particles> <n_exp> [options]

Arguments:
    n_particles    Number of SMC particles to use.
    n_exp          Number of experiments to perform.
    
Options:
    -h             Show this screen and exit.
    --plot         Plots the results.
"""

## FEATURES ####################################################################

from __future__ import absolute_import, print_function, unicode_literals

## SCRIPT ######################################################################

if __name__ == "__main__":

    ## IMPORTS ##

    from qinfer._lib import docopt
    from qinfer.smc import SMCUpdater
    from qinfer import SimplePrecessionModel, ExpSparseHeuristic, UniformDistribution

    import sys
    import numpy as np

    ## ARG HANDLING ##
    # Handle and unpack command-line arguments.

    args = docopt.docopt(__doc__, sys.argv[1:])

    n_particles = int(args["<n_particles>"])
    n_exp       = int(args["<n_exp>"])

    ## PLOTTING SUPPORT ##

    if args['--plot']:
        try:
            import matplotlib.pyplot as plt
        except Exception as ex:
            print(
                "Exception importing matplotlib: {}. "
                "Disabling plotting support.".format(ex)
            )
    else:
        plt = None

    if plt is not None:
        try:
            plt.style.use('ggplot')
        except:
            pass

    ## ACTUAL EXAMPLE ##

    model = SimplePrecessionModel()
    prior = UniformDistribution([0, 1])
    true_modelparams = prior.sample()

    updater = SMCUpdater(model, n_particles, prior)
    heuristic = ExpSparseHeuristic(updater)

    loss = np.empty((n_exp, ))
     
    for idx_exp in range(n_exp):
        experiment = heuristic()
        datum = model.simulate_experiment(true_modelparams, experiment)
        updater.update(datum, experiment)
        loss[idx_exp] = (updater.est_mean() - true_modelparams) ** 2

    print("Final loss: {:0.2e}".format(loss[-1]))

    # Now, if we've been asked to plot, go on and make a figure.
    if plt is not None:
        plt.semilogy(loss)
        plt.show()
    
