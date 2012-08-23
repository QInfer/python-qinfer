#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# test_models.py: Simple models for testing inference engines.
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

## FEATURES ##

from __future__ import division # Ensures that a/b is always a float.

## IMPORTS ##

import numpy as np

from abstract_model import Model
    
## CLASSES ##

class SimplePrecessionModel(Model):
    
    ## PROPERTIES ##
    
    @property
    def n_modelparams(self):
        return 1
        
    @property
    def expparams_dtype(self):
        return 'float'
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if the number of outcomes for each
        experiment is independent of the experiment being performed.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        return True
    
    ## METHODS ##
    
    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        return 2
    
    def likelihood(self, outcomes, modelparams, expparams):
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(SimplePrecessionModel, self).likelihood(outcomes, modelparams, expparams)
        
        # Allocating first serves to make sure that a shape mismatch later
        # will cause an error.
        pr0 = np.zeros((modelparams.shape[0], expparams.shape[0]))
        
        arg = np.dot(modelparams, expparams[..., np.newaxis].T) / 2        
        pr0 = np.cos(arg) ** 2
        
        # Now we concatenate over outcomes.
        pr0 = pr0[np.newaxis, ...]
        return np.concatenate([
            pr0 if outcomes[idx] == 0 else 1 - pr0
            for idx in xrange(outcomes.shape[0])
            ]) 

## TESTING CODE ################################################################

if __name__ == "__main__":

    m = SimplePrecessionModel()
    L = m.likelihood(
        np.array([1]),
        np.array([[0.1], [0.2], [0.4]]),
        np.array([1/2, 17/3]) * np.pi
    )
    print L
    assert m.call_count == 6
    assert L.shape == (1, 3, 2)
    
