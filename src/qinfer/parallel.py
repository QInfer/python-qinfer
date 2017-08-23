#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# parallel.py: Tools for distributing computation.
##
# © 2017, Chris Ferrie (csferrie@gmail.com) and
#         Christopher Granade (cgranade@cgranade.com).
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the copyright holder nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
##

## FEATURES ##################################################################

from __future__ import absolute_import
from __future__ import division # Ensures that a/b is always a float.

## EXPORTS ###################################################################

__all__ = ['DirectViewParallelizedModel']

## IMPORTS ###################################################################

import numpy as np
from qinfer.derived_models import DerivedModel

import warnings

try:
    import ipyparallel as ipp
    interactive = ipp.interactive
except ImportError:
    try:
        import IPython.parallel as ipp
        interactive = ipp.interactive
    except (ImportError, AttributeError):
        import warnings
        warnings.warn(
            "Could not import IPython parallel. "
            "Parallelization support will be disabled."
        )
        ipp = None
        interactive = lambda fn: fn

## LOGGING ###################################################################

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
    
## CLASSES ###################################################################

class DirectViewParallelizedModel(DerivedModel):
    r"""
    Given an instance of a :class:`Model`, parallelizes execution of that model's
    likelihood by breaking the ``modelparams`` array into segments and
    executing a segment on each member of a :class:`~ipyparallel.DirectView`.
    
    This :class:`Model` assumes that it has ownership over the DirectView, such
    that no other processes will send tasks during the lifetime of the Model.

    If you are having trouble pickling your model, consider switching to 
    ``dill`` by calling ``direct_view.use_dill()``. This mode gives more support 
    for closures.
    
    :param qinfer.Model serial_model: Model to be parallelized. This
        model will be distributed to the engines in the direct view, such that
        the model must support pickling.
    :param ipyparallel.DirectView direct_view: Direct view onto the engines
        that will be used to parallelize evaluation of the model's likelihood
        function.
    :param bool purge_client: If ``True``, then this model will purge results
        and metadata from the IPython client whenever the model cache is cleared.
        This is useful for solving memory leaks caused by very large numbers of
        calls to ``likelihood``. By default, this is disabled, since enabling
        this option can cause data loss if the client is being sent other tasks
        during the operation of this model.
    :param int serial_threshold: Sets the number of model vectors below which
        the serial model is to be preferred. By default, this is set to ``10 *
        n_engines``, where ``n_engines`` is the number of engines exposed by
        ``direct_view``.
    """
    
    ## INITIALIZER ##
    
    def __init__(self, serial_model, direct_view, purge_client=False, serial_threshold=None):
        if ipp is None:
            raise RuntimeError(
                "This model requires IPython parallelization support, "
                "but an error was raised importing IPython.parallel."
            )

        self._dv = direct_view
        self._purge_client = purge_client
        self._serial_threshold = (
            10 * self.n_engines
            if serial_threshold is None else int(serial_threshold)
        )
        
        super(DirectViewParallelizedModel, self).__init__(serial_model)
    
    ## SPECIAL METHODS ##
    
    def __getstate__(self):
        # Since instances of this class will be pickled as they are passed to
        # remote engines, we need to be careful not to include _dv
        return {
            '_underlying_model': self._underlying_model,
            '_dv': None,
            '_call_count': self._call_count,
            '_sim_count': self._sim_count,
            '_serial_threshold': self._serial_threshold
        }
    
    ## PROPERTIES ##

    # Provide _serial_model as a back-compat.
    @property
    def _serial_model(self):
        warnings.warn("_serial_model is deprecated in favor of _underlying_model.",
            DeprecationWarning
        )
        return self._underlying_model
    @_serial_model.setter
    def _serial_model(self, value):
        warnings.warn("_serial_model is deprecated in favor of _underlying_model.",
            DeprecationWarning
        )
        self._underlying_model = value
    

    @property
    def n_engines(self):
        """
        The number of engines seen by the direct view owned by this parallelized
        model.

        :rtype: int
        """
        return len(self._dv) if self._dv is not None else 0
            
    ## METHODS ##
    
    def clear_cache(self):
        """
        Clears any cache associated with the serial model and the engines
        seen by the direct view.
        """
        self.underlying_model.clear_cache()
        try:
            logger.info('DirectView results has {} items. Clearing.'.format(
                len(self._dv.results)
            ))
            self._dv.purge_results('all')
            if self._purge_client:
                self._dv.client.purge_everything()
        except:
            pass
    
    def likelihood(self, outcomes, modelparams, expparams):
        """
        Returns the likelihood for the underlying (serial) model, distributing
        the model parameter array across the engines controlled by this
        parallelized model. Returns what the serial model would return, see
        :attr:`~Model.likelihood`
        """
        # By calling the superclass implementation, we can consolidate
        # call counting there.
        super(DirectViewParallelizedModel, self).likelihood(outcomes, modelparams, expparams)

        # If there's less models than some threshold, just use the serial model.
        # By default, we'll set that threshold to be the number of engines * 10.
        if modelparams.shape[0] <= self._serial_threshold:
            return self.underlying_model.likelihood(outcomes, modelparams, expparams)
        
        if self._dv is None:
            raise RuntimeError(
                "No direct view provided; this may be because the instance was "
                "loaded from a pickle or NumPy saved array without providing a "
                "new direct view."
            )

        # Need to decorate with interactive to overcome namespace issues with
        # remote engines.
        @interactive
        def serial_likelihood(mps, sm, os, eps):
            return sm.likelihood(os, mps, eps)

        # TODO: check whether there's a better way to pass the extra parameters
        # that doesn't use so much memory.
        # The trick is that serial_likelihood will be pickled, so we need to be
        # careful about closures.
        L = self._dv.map_sync(
            serial_likelihood,
            np.array_split(modelparams, self.n_engines, axis=0),
            [self.underlying_model] * self.n_engines,
            [outcomes] * self.n_engines,
            [expparams] * self.n_engines
        )

        return np.concatenate(L, axis=1)

    def simulate_experiment(self, modelparams, expparams, repeat=1, split_by_modelparams=True):
        """
        Simulates the underlying (serial) model using the parallel 
        engines. Returns what the serial model would return, see
        :attr:`~Simulatable.simulate_experiment`

        :param bool split_by_modelparams: If ``True``, splits up
            ``modelparams`` into `n_engines` chunks and distributes 
            across engines. If ``False``, splits up ``expparams``.
        """
        # By calling the superclass implementation, we can consolidate
        # simulation counting there.
        super(DirectViewParallelizedModel, self).simulate_experiment(modelparams, expparams, repeat=repeat)

        if self._dv is None:
                raise RuntimeError(
                    "No direct view provided; this may be because the instance was "
                    "loaded from a pickle or NumPy saved array without providing a "
                    "new direct view."
                )

        # Need to decorate with interactive to overcome namespace issues with
        # remote engines.
        @interactive
        def serial_simulator(sm, mps, eps, r):
            return sm.simulate_experiment(mps, eps, repeat=r)

        if split_by_modelparams:
            # If there's less models than some threshold, just use the serial model.
            # By default, we'll set that threshold to be the number of engines * 10.
            if modelparams.shape[0] <= self._serial_threshold:
                return self.underlying_model.simulate_experiment(modelparams, expparams, repeat=repeat)

            # The trick is that serial_likelihood will be pickled, so we need to be
            # careful about closures.
            os = self._dv.map_sync(
                serial_simulator,
                [self.underlying_model] * self.n_engines,
                np.array_split(modelparams, self.n_engines, axis=0),
                [expparams] * self.n_engines,
                [repeat] * self.n_engines
            )

            return np.concatenate(os, axis=0)

        else:
            # If there's less models than some threshold, just use the serial model.
            # By default, we'll set that threshold to be the number of engines * 10.
            if expparams.shape[0] <= self._serial_threshold:
                return self.underlying_model.simulate_experiment(modelparams, expparams, repeat=repeat)

            # The trick is that serial_likelihood will be pickled, so we need to be
            # careful about closures.
            os = self._dv.map_sync(
                serial_simulator,
                [self.underlying_model] * self.n_engines,
                [modelparams] * self.n_engines,
                np.array_split(expparams, self.n_engines, axis=0),
                [repeat] * self.n_engines
            )

            return np.concatenate(os, axis=1)

