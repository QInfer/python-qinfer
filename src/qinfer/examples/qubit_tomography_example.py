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
Usage: qubit_tomography_example <n_particles> <n_exp> [options]

Arguments:
    n_particles    Number of SMC particles to use.
    n_exp          Number of experiments to perform.
    
Options:
    -h             Show this screen and exit.
    -d DIM         Dimension of the tomography problem to
                   simulate. [default: 2]
    -r RANK        Rank of states over which the prior should be
                   supported. [default: 1]
    --parallelize  If set, trials will be parallelized over available
                   cores using ipyparallel. [default: False]
    --n-trials=N   Number of simulated runs to average over. [default: 40]
"""

## FEATURES ####################################################################

from __future__ import absolute_import, print_function, division

## IMPORTS #####################################################################

from builtins import range

import numpy as np

## Imports from within QInfer. ##

from qinfer import perf_test_multiple
from qinfer.smc import SMCUpdater
from qinfer.tomography import (
    TomographyModel, gell_mann_basis, GinibreDistribution,
    RandomStabilizerStateHeuristic
)
from qinfer._lib import docopt

try:
    import qutip as qt
except ImportError:
    qt = None

## SCRIPT ######################################################################

if __name__ == "__main__":

    import sys

    if qt is None:
        print("This example requires QuTiP 3.2 or later to be installed.")
        sys.exit(1)

    import matplotlib.pyplot as plt
    
    # Handle and unpack command-line arguments.
    args = docopt.docopt(__doc__, sys.argv[1:])
    
    n_trials    = int(args["--n-trials"])
    n_particles = int(args["<n_particles>"])
    n_exp       = int(args["<n_exp>"])
    parallelize = bool(args['--parallelize'])

    dim = int(args['-d'])
    rank = int(args['-r'])

    # Model and prior initialization.
    basis = gell_mann_basis(dim)
    prior = GinibreDistribution(basis, rank=rank)
    model = TomographyModel(basis)
    heuristic_class = RandomStabilizerStateHeuristic

    # Parallelization
    if parallelize:
        try:
            import ipyparallel
            client = ipyparallel.Client()
            lbview = client.load_balanced_view()
            perf_args = {'apply': lbview.apply}
            print("Successfully connected to {} engines.".format(len(client)))

        except Exception as ex:
            print(
                "Parallelization failed with exception {}. "
                "Serially evaluating instead.".format(ex)
            )
            parallelize = False
            perf_args = {}
    else:
        perf_args = {}

    performance = perf_test_multiple(
        n_trials,
        model, n_particles, prior,
        n_exp, heuristic_class,
        progressbar=qt.ui.EnhancedTextProgressBar,
        **perf_args
    )

    plt.plot(performance['loss'].mean(axis=0))
    plt.show()
