#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# macros.py: MacroPy-based macros for writing QInfer code.
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
"""
This module exposes macros; this is not an intrinsic part of the Python
language, but can be added using import hooks. This is done by the `MacroPy`_
library.

Please note that this module is, by virtue of being an unapologetic hack,
buggy as... well, use your imagination.

.. _MacroPy: https://github.com/lihaoyi/macropy
"""

## FEATURES ##################################################################

from __future__ import absolute_import
from __future__ import division

## MACRO IMPORTS #############################################################

from macropy.core.macros import *
from macropy.core.quotes import macros, q, u, ast
macros = Macros()

## IMPORTS ###################################################################

from builtins import map

import itertools as it

import _ast

from collections import OrderedDict
    
## MACROS ####################################################################

@macros.decorator
def nop(tree, **kw):
    """
    Does nothing; useful only for debugging.
    """
    return tree

@macros.decorator
def init(tree, **kw):
    """
    Given a "class" definition of the form

    ```
    @init
    class Foo(bases):
        body
    ```

    makes a new class of the form

    ```
    class Foo(object):
        def __init__(self):
            body
    ```
    """
    old_body = tree.body
    with q as new_method:
        def __init__(self):
            pass
    new_method[0].body = old_body
    tree.body = new_method

    return tree

@macros.decorator
def modelclass(tree, args, **kw):
    """
    Given a "class" definition of the form

    ```
    @modelclass(types, validity_constraints)
    class NewModel():
        likelihood_expr
    ```

    makes a new QInfer model based on interpreting
    ``likelihood_expr`` as the probability of a
    "0" outcome in a two-outcome model.

    Any variables of named mp_* in likelihood_expr are
    interpreted as named model parameters, and will be
    appropriately vectorized. Any variables named as
    ep_* are taken to be experimental parameters, and
    will be used to construct expparams_dtype.

    Example::

        @modelclass({'t': float}, [mp_w1 > 0, mp_w2 > 0])
        class MacroMadeModel():
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
    """

    idxs = it.count()

    ## STAGE 0: METADATA ##

    # Extract the model name.
    model_name = tree.name

    # If the first element is a docstring, pull that out
    # and save it so we can inject later.
    docstring = []
    if len(tree.body) >= 1:
        if isinstance(tree.body[0], Expr):
            if isinstance(tree.body[0].value, Str):
                docstring = tree.body[0]
                del tree.body[0]
    docstring = docstring.value.s

    # Build a list of experiment parameter types.
    expparam_types = {}
    if len(args) >= 1:
        for name, val in  zip(args[0].keys, args[0].values):
            expparam_types[name.s] = val.id


    ## STAGE 1: SEARCH ##

    @Walker
    def params_search(tree, collect, **kw):
        # Don't change anything, just find model parameters.
        if type(tree) is Name:
            if tree.id.startswith('mp_'):
                mp_idx = next(idxs)
                collect((tree.id, mp_idx))
            elif tree.id.startswith('ep_'):
                name = tree.id[3:]
                if name not in expparam_types:
                    # Assume float...
                    expparam_types[name] = float

    tree, names = params_search.recurse_collect(tree)
    name_dict = OrderedDict(names)
    
    ## STAGE 2: TRANSFORM PARAMETERS ##

    @Walker
    def param_transform(tree, **kw):
        if type(tree) is Name:
            if tree.id.startswith('mp_'):
                mp_idx = name_dict[tree.id]
                tree = q[modelparams[:, u[mp_idx], np.newaxis]]
                return tree
            elif tree.id.startswith('ep_'):
                name = tree.id[3:]
                tree = q[expparams[u[name]]]
                return tree
            else:
                return tree


    tree = param_transform.recurse(tree)
    
    ## STAGE 3: GENERATE VALIDITY CONDITIONS ##

    if len(args) >= 2:
        validity_conditions = args[1]
    else:
        validity_conditions = q[[]]
    validity_conditions = param_transform.recurse(validity_conditions)

    ## STAGE 4: GENERATE RETURN ##
    # We assume that the final expression is the return value.
    # We also need to wrap this in a two-outcome rank promotion.

    tree.body[-1] = Return(q[
        Model.pr0_to_likelihood_array(outcomes,
            ast[tree.body[-1].value]
        )
    ])
    
    ## STAGE 5: GENERATE CLASS ##

    # Due to a bug in q[u[]], we can't inject expparams_dtype
    # into a u[]. Instead, let's build up the ast, bit by bit.
    expparams_dtype_ast = q[[]]
    for ep_field, ep_type in expparam_types.items():
        expparams_dtype_ast.elts.append(
            q[(u[ep_field], u[ep_type])]
        )
    
    # Try our hand at making a new class, perhaps?
    with q as new_class:
        class NewModel(Model):
            u[docstring]

            @property
            def n_modelparams(self):
                return u[len(name_dict)]

            @property
            def is_n_outcomes_constant(self):
                return True

            @property
            def modelparam_names(self):
                return u[[name[3:] for name in name_dict.keys()]]

            def n_outcomes(self, expparams):
                return 2

            @staticmethod
            def are_models_valid(modelparams):
                return np.all(ast[
                    validity_conditions
                ], axis=0)

            @property
            def expparams_dtype(self):
                return ast[expparams_dtype_ast]
            
            def likelihood(self, outcomes, modelparams, expparams):
                # FIXME: the following line breaks it.
                super(ast[Name(model_name, Load())], self).likelihood(outcomes, modelparams, expparams)


    ## STAGE 6: INJECT LIKELIHOOD ##

    to_inject = tree.body
    @Walker
    def likelihood_injector(tree, **kwargs):
        if isinstance(tree, FunctionDef):
            if tree.name == "likelihood":
                tree.body += to_inject

    new_class = likelihood_injector.recurse(new_class)

    new_class[0].name = model_name

    # Strip off the functiondef and copy over the args definition.
    tree.body = list(map(fix_missing_locations, new_class[0].body))
    tree.bases = [q[Model]]

    return tree
