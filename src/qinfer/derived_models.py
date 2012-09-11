#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# derived_models.py: Models that decorate and extend other models.
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

from __future__ import division # Ensures that a/b is always a float.

## IMPORTS #####################################################################

import numpy as np

from utils import binomial_pdf

from abstract_model import Model
    
## CLASSES #####################################################################

class BinomialModel(Model):
    """
    Model representing finite numbers of iid samples from another model,
    using the binomial distribution to calculate the new likelihood function.
    
    :param qinfer.abstract_model.Model decorated_model: An instance of a two-
        outcome model to be decorated by the binomial distribution.
        
    Note that a new experimental parameter field ``n_meas`` is added by this
    model. This parameter field represents how many times a measurement should
    be made at a given set of experimental parameters. To ensure the correct
    operation of this model, it is important that the decorated model does not
    also admit a field with the name ``n_meas``.
    """
    
    def __init__(self, decorated_model):
        super(BinomialModel, self).__init__()
        self.decorated_model = decorated_model
        
        if not (decorated_model.is_n_outcomes_constant and decorated_model.n_outcomes(None) == 2):
            raise ValueError("Decorated model must be a two-outcome model.")
        
        if isinstance(decorated_model.expparams_dtype, str):
            # We default to calling the original experiment parameters "x".
            self._expparams_scalar = True
            self._expparams_dtype = [('x', decorated_model.expparams_dtype), ('n_meas', 'uint')]
        else:
            self._expparams_scalar = False
            self._expparams_dtype = decorated_model.expparams_dtype + [('n_meas', 'uint')]
    
    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        # We have as many modelparameters as the underlying model.
        return self.decorated_model.n_modelparams
        
    @property
    def expparams_dtype(self):
        return self._expparams_dtype
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        return False
    
    ## METHODS ##
    
    def is_model_valid(self, modelparams):
        return self.decorated_model.is_model_valid(modelparams)
    
    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        return expparams['n_meas'] + 1
    
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(BinomialModel, self).likelihood(outcomes, modelparams, expparams)
        pr0 = self.decorated_model.likelihood(
            np.array([0], dtype='uint'),
            modelparams,
            expparams['x'] if self._expparams_scalar else expparams)
        
        # Now we concatenate over outcomes.
        return np.concatenate([
            binomial_pdf(expparams['n_meas'][np.newaxis, :], outcomes[idx], pr0)
            for idx in xrange(outcomes.shape[0])
            ]) 

## TESTING CODE ################################################################

if __name__ == "__main__":
    
    import operator as op
    from test_models import SimplePrecessionModel
    
    m = BinomialModel(SimplePrecessionModel())
    
    os = np.array([6, 7, 8, 9, 10])
    mps = np.array([[0.1], [0.35], [0.77]])
    eps = np.array([(0.5 * np.pi, 10), (0.51 * np.pi, 10)], dtype=m.expparams_dtype)
    
    L = m.likelihood(
        os, mps, eps
    )
    print L
    
    assert m.call_count == reduce(op.mul, [os.shape[0], mps.shape[0], eps.shape[0]]), "Call count inaccurate."
    assert L.shape == (os.shape[0], mps.shape[0], eps.shape[0]), "Shape mismatch."
    
