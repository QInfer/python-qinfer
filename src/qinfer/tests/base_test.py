#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# base_test.py: Base class for derandomized test classes.
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

## FEATURES ###################################################################

from __future__ import absolute_import
from __future__ import division # Ensures that a/b is always a float.
from future.utils import with_metaclass

## IMPORTS ####################################################################

import sys
import warnings
import abc
import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import unittest
from qinfer import Domain, Model, Simulatable, FiniteOutcomeModel, DifferentiableModel

from contextlib import contextmanager

## FUNCTIONS ##################################################################

def test_model(model, prior, expparams, stream=sys.stderr):
    """
    Tests the given Simulatable instance for errors. Useful for debugging
    new or third party models.

    :param model: Instance of Simulatable or a subclass thereof.
    :param prior: Instance of Distribution, or any other class which
        implements a function `sample` that returns valid modelparams.
    :param expparams: `np.ndarray` of experimental parameters to test with.
    :param stream: Stream to dump the results into, default is stderr.
    """

    if isinstance(model, DifferentiableModel):
        test_class = ConcreteDifferentiableModelTest
    elif isinstance(model, Model):
        test_class = ConcreteModelTest
    elif isinstance(model, Simulatable):
        test_class = ConcreteSimulatableTest
    else:
        raise ValueError("Given model has unrecognized type.")

    class TestGivenModel(test_class, DerandomizedTestCase):
        def instantiate_model(self):
            return model
        def instantiate_prior(self):
            return prior
        def instantiate_expparams(self):
            return expparams

    suite = unittest.TestLoader().loadTestsFromTestCase(TestGivenModel)
    runner = unittest.TextTestRunner(stream=stream)
    runner.run(suite)

@contextmanager
def assert_warns(category):
    """
    Context manager which asserts that its contents raise a particular
    warning.

    :param type category: Category of the warning that should be raised.
    """

    with warnings.catch_warnings(record=True) as caught_warnings:
        # Catch everything.
        warnings.simplefilter('always')

        yield

    assert any([
        issubclass(warning.category, category) for warning in caught_warnings
    ]), "No warning of category {} raised.".format(category)

## CLASSES ####################################################################

class MockModel(FiniteOutcomeModel):
    """
    Two-outcome model whose likelihood is always 0.5, irrespective of
    model parameters, outcomes or experiment parameters.
    """

    def __init__(self, n_mps=2):
        self._n_mps = n_mps
        super(MockModel, self).__init__()

    @property
    def n_modelparams(self):
        return self._n_mps

    @staticmethod
    def are_models_valid(modelparams):
        return np.ones((modelparams.shape[0], ), dtype=bool)

    @property
    def is_n_outcomes_constant(self):
        return True

    def n_outcomes(self, expparams):
        return 2


    @property
    def expparams_dtype(self):
        return [('a', float), ('b', int)]


    def likelihood(self, outcomes, modelparams, expparams):
        super(MockModel, self).likelihood(outcomes, modelparams, expparams)
        pr0 = np.ones((modelparams.shape[0], expparams.shape[0])) / 2
        return FiniteOutcomeModel.pr0_to_likelihood_array(outcomes, pr0)

class MockAsyncResult(object):
    def __init__(self, value):
        self._value = value

def MockAsyncMapResult(MockAsyncResult):
    def __iter__(self):
        return iter(self._value)

class MockDirectView(object):
    """
    Object that mocks up an ipyparallel DirectView
    using serial execution, allowing for testing of
    classes that make use of ipyparallel without needing
    to install more libraries.
    """

    n_engines = None

    def __init__(self, n_engines=1):
        self.n_engines = n_engines

    def __len__(self):
        return self.n_engines

    def clear(targets=None, block=None):
        raise NotImplementedError

    def execute(self, code, silent=True, targets=None, block=None):
        exec(code)

    def gather(self, key, dist='b', targets=None, block=None):
        raise NotImplementedError

    def get(self, key_s):
        raise NotImplementedError

    def map(self, f, *sequences, **kwargs):
        if 'block' in kwargs and kwargs['block']:
            return list(map(f, *sequences))
        else:
            return MockAsyncMapResult(list(map(f, *sequences)))

    def map_sync(self, f, *sequences):
        return self.map(f, *sequences, **dict(block=True))

    def map_async(self, f, *sequences):
        return self.map(f, *sequences, **dict(block=False))


class DerandomizedTestCase(unittest.TestCase):

    ## SETUP AND TEARDOWN ##
    # We want every test method to be setup first by seeding NumPy's random
    # number generator with a predictable seed (namely: zero). This way,
    # all of our tests are *deterministic*, and once first checked, will
    # not deviate from that behavior unless there is a change to the underlying
    # functionality.
    #
    # We do this by using the fact that nosetests and unittest both call
    # the method named "setUp" (note the capitalization!) before each
    # test method.

    def setUp(self):
        np.random.seed(0)

class ConcreteSimulatableTest(with_metaclass(abc.ABCMeta, object)):
    """
    Mixin of generic tests which can be run to test basic properties
    of any subclass of Simulatable.
    """

    # FORCED PROPERTIES ##

    # We use this abstract instantiate_* paradigm to ensure that the actual
    # property cannot change instances throughout the testing. Although
    # unlikely, this paranoid approach prevents subclasses from having
    # model return something different every time it is called!

    @abc.abstractproperty
    def instantiate_model(self):
        """
        Generates and returns an instance of the concrete Model class being tested.
        """
        pass
    @property
    def model(self):
        """
        Returns (a fixed) instance of the concrete Model class being tested.
        """
        try:
            return self._model
        except AttributeError:
            self._model = self.instantiate_model()
            return self._model

    @abc.abstractproperty
    def instantiate_prior(self):
        """
        Generates and returns a prior Distribution to be used with the model.
        """
        pass
    @property
    def prior(self):
        """
        Returns (a fixed) instance of the prior to be used while testing the model.
        """
        try:
            return self._prior
        except AttributeError:
            self._prior = self.instantiate_prior()
            return self._prior

    @abc.abstractproperty
    def instantiate_expparams(self):
        """
        Generates and returns a set of expparams to be used with the model.
        """
        pass
    @property
    def expparams(self):
        """
        Returns (a fixed) set of expparams to be used while testing the model.
        """
        try:
            return self._expparams
        except AttributeError:
            self._expparams = self.instantiate_expparams()
            return self._expparams

    ## PROPERTIES ##

    @property
    def n_expparams(self):
        """
        Number of experimental parameters to do tests with.
        """
        return self.expparams.shape[0]

    @property
    def n_models(self):
        """
        Number of model parameters to do tests with.
        """
        # Ensure that n_models is not equal to n_expparams
        return 21 if self.n_expparams == 20 else 20

    @property
    def n_outcomes(self):
        """
        Number of outcomes to do tests with.
        """
        # Ensure that this is not equal to n_models or n_expparams
        return self.n_models + self.n_expparams

    @property
    def modelparams(self):
        """
        Fixed set of model parameter to do tests with.
        """
        try:
            return self._modelparams
        except AttributeError:
            # get modelparams by sampling the prior
            mps = self.prior.sample(n=self.n_models)
            self._modelparams = mps
            return self._modelparams

    @property
    def outcomes(self):
        """
        Fixed set of outcomes to do tests with. If you have
        a weird model with different outcome dtypes, you
        may want to set this property manually.
        """
        try:
            return self._outcomes
        except AttributeError:
            # get some of our elements from the domain
            os = self.model.domain(self.expparams)[0].values
            while os.shape[0] < self.n_outcomes:
                os = np.concatenate([os,os])
            if os.shape[0] > self.n_outcomes:
                os = os[:self.n_outcomes]
            self._outcomes = os

            return self._outcomes


    ## TESTS ##

    def test_simulate_experiment(self):
        """
        Tests that simulate_experiment does not fail and has the right
        output format.
        """

        # ensure that repeat is not equal to n_models or n_expparams
        repeat = 2
        while repeat == self.n_expparams or repeat == self.n_models:
            repeat = repeat + 1

        outcomes = self.model.simulate_experiment(self.modelparams, self.expparams, repeat=repeat)

        assert(outcomes.shape == (
            repeat,
            self.n_models,
            self.n_expparams)
            )

        # check that outcomes are in the right domains
        for idx_ep in range(self.n_expparams):
            domain = self.model.domain(self.expparams[idx_ep:idx_ep+1])[0]
            assert(domain.in_domain(outcomes[:,:,idx_ep].flatten()))


    def test_update_timestep(self):
        """
        Tests that update_timstep does not fail and
        has the right output format.
        """

        mps = self.model.update_timestep(self.modelparams, self.expparams)

        assert(mps.shape == (
                self.n_models,
                self.model.n_modelparams,
                self.n_expparams
            ))
        mps = mps.transpose((2,0,1)).reshape(self.n_models * self.n_expparams, -1)
        
        assert(np.all(self.model.are_models_valid(mps)))

    def test_domain_with_none(self):
        """
        Tests that the domain property of a Model works with the None input
        whenever is_n_outcomes_constant is True.
        """
        if self.model.is_n_outcomes_constant:
            domain = self.model.domain(None)
            assert(isinstance(domain, Domain))


    def test_domain(self):
        """
        Tests that the domain property returns a list of
        domains of the correct length
        """
        domains = self.model.domain(self.expparams)
        assert(len(domains) == self.n_expparams)
        for domain in domains:
            assert(isinstance(domain, Domain))

class ConcreteModelTest(ConcreteSimulatableTest):
    """
    Mixin of generic tests which can be run to test basic properties
    of any subclass of Model.
    """

    ## TESTS ##

    def test_are_models_valid(self):
        """
        Tests that are_models_valid does not fail.
        """
        # we are more interested in whether this fails than if the models are valid
        self.model.are_models_valid(self.modelparams)

    def test_canonicalize(self):
        """
        Tests that canonicalize does not fail and that it
        returns valid models for the tester's specific modelparams.
        """

        new_mps = self.model.canonicalize(self.modelparams)
        assert(np.all(self.model.are_models_valid(new_mps)))


    def test_likelihood(self):
        """
        Tests that likelihood does not fail and has the right
        output format.
        """

        L = self.model.likelihood(self.outcomes, self.modelparams, self.expparams)

        assert(L.shape == (
            self.n_outcomes,
            self.n_models,
            self.n_expparams)
            )

class ConcreteDifferentiableModelTest(ConcreteModelTest):
    """
    Mixin of generic tests which can be run to test basic properties
    of any subclass of Model.
    """

    ## TESTS ##

    def test_fisher_information(self):
        """
        Tests that fisher information does not fail and has the right
        output format.
        """

        fisher = self.model.fisher_information(self.modelparams, self.expparams)

        assert(fisher.shape == (
            self.model.n_modelparams,
            self.model.n_modelparams,
            self.n_models,
            self.n_expparams))

    def test_score(self):
        """
        Tests that score does not fail and has the right
        output format.
        """

        score1 = self.model.score(self.outcomes, self.modelparams, self.expparams, return_L=False)
        L1 = self.model.likelihood(self.outcomes, self.modelparams, self.expparams)
        score, L = self.model.score(self.outcomes, self.modelparams, self.expparams, return_L=True)

        # Ensure some consistency
        assert_almost_equal(score1, score, 3)
        assert_almost_equal(L1, L, 3)

        # Dimensions must be correct
        assert(score.shape == (
            self.model.n_modelparams,
            self.n_outcomes,
            self.n_models,
            self.n_expparams)
            )

class ConcreteDomainTest(with_metaclass(abc.ABCMeta, object)):
    """
    Mixin of generic tests which can be run to test basic properties
    of any subclass of Domain.
    """

    # FORCED PROPERTIES ##

    # We use this abstract instantiate_* paradigm to ensure that the actual
    # property cannot change instances throughout the testing.

    @abc.abstractproperty
    def instantiate_domain(self):
        """
        Generates and returns an instance of the concrete Domain class being tested.
        """
        pass
    @property
    def domain(self):
        """
        Returns (a fixed) instance of the concrete Model class being tested.
        """
        try:
            return self._domain
        except AttributeError:
            self._domain = self.instantiate_domain()
            return self._domain

    @abc.abstractproperty
    def instantiate_good_values(self):
        """
        Returns a list of values in the domain.
        """
        pass
    @property
    def good_values(self):
        """
        Returns (a fixed) list of values in the domain.
        """
        try:
            return self._good_values
        except AttributeError:
            self._good_values = self.instantiate_good_values()
            return self._good_values

    @abc.abstractproperty
    def instantiate_bad_values(self):
        """
        Returns a list of values not in the domain.
        """
        pass
    @property
    def bad_values(self):
        """
        Returns (a fixed) list of values not in the domain.
        """
        try:
            return self._bad_values
        except AttributeError:
            self._bad_values = self.instantiate_bad_values()
            return self._bad_values


    ## TESTS ##

    def test_is_cts_or_is_descrete(self):
        """
        Tests that is_continuous is not is_discrete
        """
        assert(self.domain.is_continuous or not self.domain.is_continuous)
        assert(self.domain.is_continuous is not self.domain.is_discrete)


    def test_is_finite(self):
        """
        Tests that is_finite is bool and consistent
        """

        assert(self.domain.is_finite or not self.domain.is_finite)
        if self.domain.is_finite:
            assert(self.domain.is_discrete)

    def test_example_point(self):
        """
        Tests that the example point is in the domain and has the right dtype
        """
        assert(self.domain.in_domain(self.domain.example_point))
        assert_equal(self.domain.example_point, self.domain.example_point.astype(self.domain.dtype))


    def test_values(self):
        """
        Tests that n_members is consistent
        """
        values = self.domain.values
        if self.domain.n_members < np.inf:
            assert(values.size == self.domain.n_members)
        assert(self.domain.in_domain(values))

    def test_in_domain(self):
        """
        Tests that good_values are in the domain and bad_values are not.
        (self.values is tested elsewhere)
        """
        for v in self.good_values:
            try:
                assert(self.domain.in_domain(v))
            except AssertionError as e:
                e.args += ('Current good value: {}'.format(v),)
                raise e
        for v in self.bad_values:
            try:
                assert(not self.domain.in_domain(v))
            except AssertionError as e:
                e.args += ('Current bad value: {}'.format(v),)
                raise e
