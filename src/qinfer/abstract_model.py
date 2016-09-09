#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# abstract_model.py: Abstract interfaces for models with different levels of
#     functionality.
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

## FEATURES ##################################################################

from __future__ import absolute_import
from __future__ import division, unicode_literals

## EXPORTS ###################################################################

__all__ = [
    'Simulatable',
    'Model',
    'FiniteOutcomeModel',
    'DifferentiableModel'
]

## IMPORTS ###################################################################

from builtins import range, map
from future.utils import with_metaclass

import abc
    # Python standard library package for specifying abstract classes.
import numpy as np
import warnings

from qinfer.utils import safe_shape
from qinfer.domains import IntegerDomain
    
## CLASSES ###################################################################

class Simulatable(with_metaclass(abc.ABCMeta, object)):
    r"""
    Represents a system which can be simulated according to
    various model parameters and experimental control parameters
    in order to produce representative data.

    See :ref:`models_guide` for more details.

    :param bool allow_identical_outcomes: Whether the method ``outcomes`` should 
        be allowed to return multiple identical outcomes for a given ``expparam``.
        It will be more efficient to set this to ``True`` whenever it is likely 
        that multiple identical outcomes will occur.
    
    """

    def __init__(self):
        """
        :param bool always_resample_outcomes: Resample outcomes stochastically with 
            each outcome call.
        :param :class:`~numpy.ndarray` initial_outcomes: Initial set of outcomes 
            that may be supplied. Otherwise initial outcomes default to 
            zeros. 
        """
        self._sim_count = 0
        
        # Initialize a default scale matrix.
        self._Q = np.ones((self.n_modelparams,))
        
    ## ABSTRACT PROPERTIES ##
    
    @abc.abstractproperty
    def n_modelparams(self):
        """
        Returns the number of real model parameters admitted by this model.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a :class:`Model` instance.
        """
        pass
        
    @abc.abstractproperty
    def expparams_dtype(self):
        """
        Returns the dtype of an experiment parameter array. For a
        model with single-parameter control, this will likely be a scalar dtype,
        such as ``"float64"``. More generally, this can be an example of a
        record type, such as ``[('time', py.'float64'), ('axis', 'uint8')]``.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        pass

            
    ## CONCRETE PROPERTIES ##
    
    @property
    def is_n_outcomes_constant(self):
        """
        Returns ``True`` if and only if both the domain and ``n_outcomes``
        are independent of the expparam.
        
        This property is assumed by inference engines to be constant for
        the lifetime of a Model instance.
        """
        return True

    @property
    def model_chain(self):
        """
        Returns a tuple of models upon which this model is based,
        such that properties and methods of underlying models for
        models that decorate other models can be accessed. For a
        standalone model, this is always the empty tuple.
        """
        return ()

    @property
    def base_model(self):
        """
        Returns the most basic model that this model depends on.
        For standalone models, this property satisfies ``model.base_model is model``.
        """
        return self

    @property
    def underlying_model(self):
        """
        Returns the model that this model is based on (decorates)
        if such a model exists, or ``None`` if this model is
        independent.
        """
        return self.model_chain[-1] if self.model_chain else None
    
    @property
    def sim_count(self):
        """
        Returns the number of data samples that have been produced by
        this simulator.

        :rtype: int
        """
        return self._sim_count
        
    @property
    def Q(self):
        r"""
        Returns the diagonal of the scale matrix :math:`\matr{Q}` that
        relates the scales of each of the model parameters. In particular,
        the quadratic loss for this Model is defined as:
        
        .. math::
            L_{\matr{Q}}(\vec{x}, \hat{\vec{x}}) =
            (\vec{x} - \hat{\vec{x}})^\T \matr{Q} (\vec{x} - \hat{\vec{x}})

        If a subclass does not explicitly define the scale matrix, it is taken
        to be the identity matrix of appropriate dimension.
        
        :return: The diagonal elements of :math:`\matr{Q}`.
        :rtype: :class:`~numpy.ndarray` of shape ``(n_modelparams, )``.
        """
        return self._Q
        
    @property
    def modelparam_names(self):
        """
        Returns the names of the various model parameters admitted by this
        model, formatted as LaTeX strings.
        """
        return list(map("x_{{{}}}".format, range(self.n_modelparams)))

    ## CONCRETE METHODS ##

    def _repr_html_(self, suppress_base=False):
        s = r"""
            <strong>{type.__name__}</strong> at 0x{id:0x}: {n_mp} model parameter{plural}
        """.format(
            id=id(self), type=type(self),
            n_mp=self.n_modelparams,
            plural="" if self.n_modelparams == 1 else "s"
        )
        if not suppress_base and self.model_chain:
            s += r"""<br>
            <p>Model chain:</p>
            <ul>{}
            </ul>
            """.format(r"\n".join(
                u"<li>{}</li>".format(model._repr_html_(suppress_base=True))
                for model in reversed(self.model_chain)
            ))
        return s

    def are_expparam_dtypes_consistent(self, expparams):
        """
        Returns ``True`` iff all of the given expparams 
        correspond to outcome domains with the same dtype.
        For efficiency, concrete subclasses should override this method 
        if the result is always ``True``.

        :param np.ndarray expparams: Array of expparamms 
             of type ``expparams_dtype``
        :rtype: ``bool``
        """
        if self.is_n_outcomes_constant:
            # This implies that all domains are equal, so this must be true
            return True

        # otherwise we have to actually check all the dtypes
        if expparams.size > 0:
            domains = self.domain(expparams)
            first_dtype = domains[0].dtype
            return all(domain.dtype == first_dtype for domain in domains[1:])
        else:
            return True
    
    ## ABSTRACT METHODS ##
    
    @abc.abstractmethod
    def n_outcomes(self, expparams):
        """
        Returns an array of dtype ``uint`` describing the number of outcomes
        for each experiment specified by ``expparams``.
        If the number of outcomes does not depend on expparams (i.e. 
        ``is_n_outcomes_constant`` is ``True``), this method 
        should return a single number.
        If there are an infinite (or intractibly large) number of outcomes, 
        this value specifies the number of outcomes to randomly sample.
        
        :param numpy.ndarray expparams: Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property.
        """
        pass
    
    @abc.abstractmethod
    def domain(self, exparams):
        """
        Returns a list of :class:`Domain` objects, one for each input expparam.

        :param numpy.ndarray expparams:  Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property, or, in the case where ``n_outcomes_constant`` is ``True``,
            ``None`` should be a valid input.

        :rtype: list of :class:`Domain`
        """
        pass

    @abc.abstractmethod
    def are_models_valid(self, modelparams):
        """
        Given a shape ``(n_models, n_modelparams)`` array of model parameters,
        returns a boolean array of shape ``(n_models)`` specifying whether
        each set of model parameters represents is valid under this model.
        """
        pass
        
    @abc.abstractmethod
    def simulate_experiment(self, modelparams, expparams, repeat=1):
        """
        Produces data according to the given model parameters and experimental
        parameters, structured as a NumPy array.

        :param np.ndarray modelparams: A shape ``(n_models, n_modelparams)``
            array of model parameter vectors describing the hypotheses under
            which data should be simulated.
        :param np.ndarray expparams: A shape ``(n_experiments, )`` array of
            experimental control settings, with ``dtype`` given by 
            :attr:`~qinfer.Model.expparams_dtype`, describing the
            experiments whose outcomes should be simulated.
        :param int repeat: How many times the specified experiment should
            be repeated.
        :rtype: np.ndarray
        :return: A three-index tensor ``data[i, j, k]``, where ``i`` is the repetition,
            ``j`` indexes which vector of model parameters was used, and where
            ``k`` indexes which experimental parameters where used. If ``repeat == 1``,
            ``len(modelparams) == 1`` and ``len(expparams) == 1``, then a scalar
            datum is returned instead.
        
        """
        self._sim_count += modelparams.shape[0] * expparams.shape[0] * repeat
        assert(self.are_expparam_dtypes_consistent(expparams))

    ## CONCRETE METHODS ##
    
    def clear_cache(self):
        """
        Tells the model to clear any internal caches used in computing
        likelihoods and drawing samples. Calling this method should not cause
        any different results, but should only affect performance.
        """
        # By default, no cache to clear.
        pass
    
    def experiment_cost(self, expparams):
        """
        Given an array of experimental parameters, returns the cost associated
        with performing each experiment. By default, this cost is constant
        (one) for every experiment.
        
        :param expparams: An array of experimental parameters for which the cost
            is to be evaluated.
        :type expparams: :class:`~numpy.ndarray` of ``dtype`` given by
            :attr:`~Model.expparams_dtype`
        :return: An array of costs corresponding to the specified experiments.
        :rtype: :class:`~numpy.ndarray` of ``dtype`` ``float`` and of the
            same shape as ``expparams``.
        """
        return np.ones(expparams.shape)
        
    def distance(self, a, b):
        r"""
        Gives the distance between two model parameter vectors :math:`\vec{a}` and
        :math:`\vec{b}`. By default, this is the vector 1-norm of the difference
        :math:`\mathbf{Q} (\vec{a} - \vec{b})` rescaled by
        :attr:`~Model.Q`.
        
        :param np.ndarray a: Array of model parameter vectors having shape
            ``(n_models, n_modelparams)``.
        :param np.ndarray b: Array of model parameters to compare to, having
            the same shape as ``a``.
        :return: An array ``d`` of distances ``d[i]`` between ``a[i, :]`` and
            ``b[i, :]``.
        """
        
        return np.apply_along_axis(
            lambda vec: np.linalg.norm(vec, 1),
            1,
            self.Q * (a - b)
        )
        
    def update_timestep(self, modelparams, expparams):
        r"""
        Returns a set of model parameter vectors that is the update of an
        input set of model parameter vectors, such that the new models are
        conditioned on a particular experiment having been performed.
        By default, this is the trivial function
        :math:`\vec{x}(t_{k+1}) = \vec{x}(t_k)`.
        
        :param np.ndarray modelparams: Set of model parameter vectors to be
            updated.
        :param np.ndarray expparams: An experiment parameter array describing
            the experiment that was just performed.
        
        :return np.ndarray: Array of shape
            ``(n_models, n_modelparams, n_experiments)`` describing the update
            of each model according to each experiment.
        """
        return np.tile(modelparams, (expparams.shape[0],1,1)).transpose((1,2,0))

    def canonicalize(self, modelparams):
        r"""
        Returns a canonical set of model parameters corresponding to a given
        possibly non-canonical set. This is used for models in which there
        exist model parameters :math:`\vec{x}_i` and :\math:`\vec{x}_j` such
        that

        .. math::

            \Pr(d | \vec{x}_i; \vec{e}) = \Pr(d | \vec{x}_j; \vec{e})

        for all outcomes :math:`d` and experiments :math:`\vec{e}`. For
        models admitting such an ambiguity, this
        method should then be overridden to return a consistent choice
        out of such vectors, hence avoiding supurious model degeneracies.

        Note that, by default, :class:`~qinfer.smc.SMCUpdater` will *not*
        call this method.
        """
        return modelparams

class Model(Simulatable):
    """
    Represents a system which can be simulated according to
    various model parameters and experimental control parameters
    in order to produce the probability of a hypothetical data
    record. As opposed to :class:`~qinfer.Simulatable`, instances
    of :class:`~qinfer.Model` not only produce data consistent
    with the description of a system, but also evaluate the probability
    of that data arising from the system.

    :param bool allow_identical_outcomes: Whether the method 
        ``representative_outcomes`` should be allowed to return multiple 
        identical outcomes for a given ``expparam``.
    :param float outcome_warning_threshold: Threshold value below which 
        ``representative_outcomes`` 
        will issue a warning about the representative outcomes 
        not adequately covering the domain with respect to 
        the relevant distribution.

    See :ref:`models_guide` for more details.
    """

    ## INITIALIZERS ##
    def __init__(self, allow_identical_outcomes=False,
            outcome_warning_threshold=0.99):
        super(Model, self).__init__()
        self._call_count = 0
        self._allow_identical_outcomes = allow_identical_outcomes
        self._outcome_warning_threshold = outcome_warning_threshold

    ## CONCRETE PROPERTIES ##

    @property
    def call_count(self):
        """
        Returns the number of points at which
        the probability of this model has been evaluated,
        where a point consists of a hypothesis about the model (a vector of
        model parameters), an experimental control setting (expparams) and
        a hypothetical or actual datum.
        :rtype: int
        """
        return self._call_count

    ## ABSTRACT METHODS ##

    @abc.abstractmethod
    def likelihood(self, outcomes, modelparams, expparams):
        r"""
        Calculates the probability of each given outcome, conditioned on each
        given model parameter vector and each given experimental control setting.

        :param np.ndarray modelparams: A shape ``(n_models, n_modelparams)``
            array of model parameter vectors describing the hypotheses for
            which the likelihood function is to be calculated.
        :param np.ndarray expparams: A shape ``(n_experiments, )`` array of
            experimental control settings, with ``dtype`` given by 
            :attr:`~qinfer.Simulatable.expparams_dtype`, describing the
            experiments from which the given outcomes were drawn.
            
        :rtype: np.ndarray
        :return: A three-index tensor ``L[i, j, k]``, where ``i`` is the outcome
            being considered, ``j`` indexes which vector of model parameters was used,
            and where ``k`` indexes which experimental parameters where used.
            Each element ``L[i, j, k]`` then corresponds to the likelihood
            :math:`\Pr(d_i | \vec{x}_j; e_k)`.
        """
        
        # Count the number of times the inner-most loop is called.
        self._call_count += (
            safe_shape(outcomes) * safe_shape(modelparams) * safe_shape(expparams)
        )

    @property
    def allow_identical_outcomes(self):
        """
        Whether the method ``representative_outcomes`` should be allowed to return multiple 
        identical outcomes for a given ``expparam``.
        It will be more efficient to set this to ``True`` whenever it is likely 
        that multiple identical outcomes will occur.

        :return: Flag state.
        :rtype: ``bool``
        """
        return self._allow_identical_outcomes
    @allow_identical_outcomes.setter
    def allow_identical_outcomes(self, value):
        """
        Whether the method ``representative_outcomes`` should be allowed to return multiple 
        identical outcomes for a given ``expparam``.
        It will be more efficient to set this to ``True`` whenever it is likely 
        that multiple identical outcomes will occur.

        :param bool allow_identical_outcomes: Value of flag.
        """
        self._allow_identical_outcomes = value

    @property
    def outcome_warning_threshold(self):
        """
        Threshold value below which ``representative_outcomes`` 
        will issue a warning about the representative outcomes 
        not adequately covering the domain with respect to 
        the relevant distribution.

        :return: Threshold value.
        :rtype: ``float``
        """
        return self._outcome_warning_threshold
    @outcome_warning_threshold.setter
    def outcome_warning_threshold(self, value):
        """
        Threshold value below which ``representative_outcomes`` 
        will issue a warning about the representative outcomes 
        not adequately covering the domain with respect to 
        the relevant distribution.

        :param float value: Threshold value.
        """
        self._outcome_warning_threshold = value
    
                
    ## CONCRETE METHODS ##
    # These methods depend on the abstract methods, and thus their behaviors
    # change in each inheriting class.
    
    def is_model_valid(self, modelparams):
        """
        Returns True if and only if the model parameters given are valid for
        this model.
        """
        return self.are_models_valid(modelparams[np.newaxis, :])[0]
    
            
class LinearCostModelMixin(Model):
    # FIXME: move this mixin to a new module.
    # TODO: test this mixin.
    """
    This mixin implements :meth:`Model.experiment_cost` by setting the
    cost of an experiment equal to the value of a given field of each
    ``expparams`` element (by default, ``t``).
    """
    _field = "t"
    
    def experiment_cost(self, expparams):
        return expparams[self._field]

class FiniteOutcomeModel(Model):
    """
    Represents a system in the same way that :class:`~qinfer.Model`,
    except that it is demanded that the number of outcomes for any 
    experiment be known and finite.

    :param bool allow_identical_outcomes: Whether the method 
        ``representative_outcomes`` should be allowed to return multiple 
        identical outcomes for a given ``expparam``.
    :param float outcome_warning_threshold: Threshold value below which 
        ``representative_outcomes`` 
        will issue a warning about the representative outcomes 
        not adequately covering the domain with respect to 
        the relevant distribution.
    :param int n_outcomes_cutoff: If ``n_outcomes`` exceeds this value, 
        ``representative_outcomes`` will use this 
        value in its place. This is useful in the case
        of a finite yet untractible number of outcomes. Use ``None`` 
        for no cutoff.

    See :class:`~qinfer.Model` and :ref:`models_guide` for more details.
    """
    
    ## INITIALIZERS ##
    def __init__(self, 
            allow_identical_outcomes=False, 
            outcome_warning_threshold=0.99, 
            n_outcomes_cutoff=None
    ):
        super(FiniteOutcomeModel, self).__init__(
            outcome_warning_threshold=outcome_warning_threshold, 
            allow_identical_outcomes=allow_identical_outcomes)
        self._n_outcomes_cutoff = n_outcomes_cutoff

        if self.is_n_outcomes_constant:
            # predefine if we can
            self._domain = IntegerDomain(min=0,max=self.n_outcomes(None)-1)
    
    ## CONCRETE PROPERTIES ##

    @property
    def n_outcomes_cutoff(self):
        """
        If ``n_outcomes`` exceeds this value for 
        some expparm, ``representative_outcomes`` will use this 
        value in its place. This is useful in the case
        of a finite yet untractible number of outcomes.

        :return: Cutoff value.
        :rtype: ``int``
        """
        return self._n_outcomes_cutoff
    @n_outcomes_cutoff.setter
    def n_outcomes_cutoff(self, value):
        """
        If ``n_outcomes`` exceeds this value, 
        ``representative_outcomes`` will use this 
        value in its place. This is useful in the case
        of a finite yet untractible number of outcomes.

        :param int value: Cutoff value.
        """
        self.n_outcomes_cutoff = value
   
    ## ABSTRACT METHODS ##
                
    ## CONCRETE METHODS ##
    # These methods depend on the abstract methods, and thus their behaviors
    # change in each inheriting class.

    def domain(self, expparams):
        """
        Returns a list of :class:`Domain` objects, one for each input expparam.

        :param numpy.ndarray expparams:  Array of experimental parameters. This
            array must be of dtype agreeing with the ``expparams_dtype``
            property, or, in the case where ``n_outcomes_constant`` is ``True``,
            ``None`` should be a valid input.

        :rtype: list of ``Domain``
        """
        # As a convenience to most users, we define domain for them. If a 
        # fancier domain is desired, this method can easily be overridden.
        if self.is_n_outcomes_constant:
            return self._domain if expparams is None else [self._domain for ep in expparams]
        else:
            return [IntegerDomain(min=0,max=n_o-1) for n_o in self.n_outcomes(expparams)]

    def simulate_experiment(self, modelparams, expparams, repeat=1):
        # Call the superclass simulate_experiment, not recording the result.
        # This is used to count simulation calls.
        super(FiniteOutcomeModel, self).simulate_experiment(modelparams, expparams, repeat)
        
        if self.is_n_outcomes_constant:
            # In this case, all expparams have the same domain
            all_outcomes = self.domain(None).values
            probabilities = self.likelihood(all_outcomes, modelparams, expparams)
            cdf = np.cumsum(probabilities, axis=0)
            randnum = np.random.random((repeat, 1, modelparams.shape[0], expparams.shape[0]))
            outcome_idxs = all_outcomes[np.argmax(cdf > randnum, axis=1)]
            outcomes = all_outcomes[outcome_idxs]
        else:
            # Loop over each experiment, sadly.
            # Assume all domains have the same dtype
            assert(self.are_expparam_dtypes_consistent(expparams))
            dtype = self.domain(expparams[0, np.newaxis])[0].dtype
            outcomes = np.empty((repeat, modelparams.shape[0], expparams.shape[0]), dtype=dtype)
            for idx_experiment, single_expparams in enumerate(expparams[:, np.newaxis]):
                all_outcomes = self.domain(single_expparams).values
                probabilities = self.likelihood(all_outcomes, modelparams, single_expparams)
                cdf = np.cumsum(probabilities, axis=0)[..., 0]
                randnum = np.random.random((repeat, 1, modelparams.shape[0]))
                outcomes[:, :, idx_experiment] = all_outcomes[np.argmax(cdf > randnum, axis=1)]
                
        return outcomes[0, 0, 0] if repeat == 1 and expparams.shape[0] == 1 and modelparams.shape[0] == 1 else outcomes


    ## STATIC METHODS ##
    # These methods are provided as a convienence to make it easier to write
    # simple models.
    
    @staticmethod
    def pr0_to_likelihood_array(outcomes, pr0):
        """
        Assuming a two-outcome measurement with probabilities given by the
        array ``pr0``, returns an array of the form expected to be returned by
        ``likelihood`` method.
        
        :param numpy.ndarray outcomes: Array of integers indexing outcomes.
        :param numpy.ndarray pr0: Array of shape ``(n_models, n_experiments)``
            describing the probability of obtaining outcome ``0`` from each
            set of model parameters and experiment parameters.
        """
        pr0 = pr0[np.newaxis, ...]
        pr1 = 1 - pr0

        if len(np.shape(outcomes)) == 0:
            outcomes = np.array(outcomes)[None]
                    
        return np.concatenate([
            pr0 if outcomes[idx] == 0 else pr1
            for idx in range(safe_shape(outcomes))
        ]) 
        
class DifferentiableModel(with_metaclass(abc.ABCMeta, Model)):
    
    @abc.abstractmethod
    def score(self, outcomes, modelparams, expparams, return_L=False):
        r"""
        Returns the score of this likelihood function, defined as:
        
        .. math::
        
            q(d, \vec{x}; \vec{e}) = \vec{\nabla}_{\vec{x}} \log \Pr(d | \vec{x}; \vec{e}).
            
        Calls are represented as a four-index tensor
        ``score[idx_modelparam, idx_outcome, idx_model, idx_experiment]``.
        The left-most index may be suppressed for single-parameter models.
        
        If return_L is True, both `q` and the likelihood `L` are returned as `q, L`.
        """
        pass
        
    def fisher_information(self, modelparams, expparams):
        """
        Returns the covariance of the score taken over possible outcomes,
        known as the Fisher information.
        
        The result is represented as the four-index tensor
        ``fisher[idx_modelparam_i, idx_modelparam_j, idx_model, idx_experiment]``,
        which gives the Fisher information matrix for each model vector
        and each experiment vector.
        
        .. note::
            
            The default implementation of this method calls
            :meth:`~DifferentiableModel.score()` for each possible outcome,
            which can be quite slow. If possible, overriding this method can
            give significant speed advantages.
        """
        
        if self.is_n_outcomes_constant:
            outcomes = np.arange(self.n_outcomes(expparams))
            scores, L = self.score(outcomes, modelparams, expparams, return_L=True)
            
            assert len(scores.shape) in (3, 4)
            
            if len(scores.shape) == 3:
                scores = scores[np.newaxis, :, :, :]
            
            # Note that E[score] = 0 by regularity assumptions, so we only
            # need the expectation over the outer product.
            return np.einsum("ome,iome,jome->ijme",
                L, scores, scores
            )
        else:
            # Indexing will be a major pain here, so we need to start
            # by making an empty array, so that index errors will be raised
            # when (not if!) we make mistakes.
            fisher = np.empty((
                self.n_modelparams, self.n_modelparams,
                modelparams.shape[0], expparams.shape[0]
            ))
            
            # Now we loop over experiments, since we cannot vectorize the
            # expectation value over data.
            for idx_experiment, experiment in enumerate(expparams):
                experiment = experiment.reshape((1,))
                n_o = self.n_outcomes(experiment)
            
                outcomes = np.arange(n_o)
                scores, L = self.score(outcomes, modelparams, experiment, return_L=True)
                
                fisher[:, :, :, idx_experiment] = np.einsum("ome,iome,jome->ijme",
                    L, scores, scores
                )
            
            return fisher
            
            
