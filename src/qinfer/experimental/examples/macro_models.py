#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# macro_models.py: Implements likelihood models using MacroPy.
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

## FEATURES ##################################################################

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

## ACTIVATION GUARD ##########################################################

if __name__ == "__main__":
    print (
        "MacroPy modules cannot be run directly, but must be imported by a "
        "plain Python module. To run this example: \n"
        ">>> import macropy.activate\n"
        ">>> import qinfer.experimental.examples.macro_models as mm\n"
        ">>> mm.main()\n"
        "Alternatively, run the run_macro_models module:\n"
        "$ python -m qinfer.experimental.examples.run_macro_models"
    )
    import sys
    sys.exit(1)

## MACRO IMPORTS #############################################################

from macropy.core.quotes import macros, q
from macropy.tracing import macros, show_expanded
from qinfer.experimental.macros import macros, modelclass

## IMPORTS ###################################################################

import numpy as np
from qinfer import Model, UniformDistribution
from qinfer.smc import SMCUpdater

## MODELS ####################################################################

@modelclass({'t': float}, [mp_w1 > 0, mp_w2 > 0])
class SimpleMacroModel():
    '''
    This will be turned into a model with two
    model parameters, w1 and w2, and with one
    expeirmental parameter, t.

    Valid models must have w1 > 0 and w2 > 0.
    '''
    dw = mp_w1 - mp_w2
    # Note that there is no return statement
    # here; the last line is returned implicitly.
    np.cos(dw * ep_t) ** 2

@modelclass({'t': float, 'wq': float}, [])
class SSKModel():
    """
    Implements the no-decoherence model of [SSK14]_.
    """
    dW = ep_wq - mp_wr
    wR = np.sqrt(dW**2 + 4 * mp_g**2)

    (1 / 2) * (
        ((4 * mp_g**2) / wR**2) * np.cos(
            wR * ep_t
        ) +
        1 +
        (dW **2 / wR**2)
    )

## MAIN ######################################################################

def main():
    m = SimpleMacroModel()
    prior = UniformDistribution([[0, 1], [0, 1]])
    u = SMCUpdater(m, 1000, prior)
    modelparams = prior.sample()
    expparams = np.array([(12.0,)], dtype=m.expparams_dtype)
    datum = m.simulate_experiment(modelparams, expparams)
    print(datum)
    u.update(datum, expparams)
    print(u.est_mean())
    print(m.call_count)
