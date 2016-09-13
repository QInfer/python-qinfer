..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _models_guide:
    
.. currentmodule:: qinfer

Designing and Using Models
==========================

Introduction
------------

The concept of a **model** is key to the use of QInfer. A model defines the
probability distribution over experimental data given hypotheses about the
system of interest, and given a description of the measurement performed. This
distribution is called the *likelihood function*, and it encapsulates the
definition of the model.

In QInfer, likelihood functions are represented as classes inheriting from either
:class:`Model`, when the likelihood function can be numerically
evaluated, or :class:`Simulatable` when only samples from the
function can be efficiently generated.

Using Models and Simulations
----------------------------

Basic Functionality
~~~~~~~~~~~~~~~~~~~

Both :class:`Model` and :class:`Simulatable` offer
basic functionality to describe how they are parameterized, what outcomes are
possible, etc. For this example, we will use a premade model,
:class:`~qinfer.SimplePrecessionModel`. This model implements the likelihood
function

.. math::

    \Pr(d | \omega; t) = \begin{cases}
        \cos^2 (\omega t / 2) & d = 0 \\
        \sin^2 (\omega t / 2) & d = 1
    \end{cases},

as can be derived from Born's Rule for a spin-½ particle prepared and measured
in the
:math:`\left|+\right\rangle\propto\left|0\right\rangle+\left|1\right\rangle` state, and evolved under :math:`H = \omega \sigma_z / 2` for some time
:math:`t`.

In this way, we see that by defining the likelihood function in terms of the
hypothetical outcome :math:`d`, the model parameter :math:`\omega`, and the experimental
parameter :math:`t`, we can reason about the experimental data that we would extract
from the system.

In order to use this likelihood function, we must instantiate the model that
implements the likelihood. Since :class:`SimplePrecessionModel` is
provided with QInfer, we can simply import it and make an instance.

>>> from qinfer import SimplePrecessionModel
>>> m = SimplePrecessionModel()

Once a model or simulator has been created, you can query how many model
parameters it admits and how many outcomes a given experiment can have.

>>> print(m.n_modelparams)
1
>>> print(m.modelparam_names)
['\\omega']
>>> print(m.is_n_outcomes_constant)
True
>>> print(m.n_outcomes(expparams=0))
2

Model and Experiment Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The division between unknown parameters that we are trying to learn (:math:`\omega`
in the :class:`SimplePrecessionModel` example) and the controls that we can use to
design measurements (:math:`t`) is generic, and is key to how QInfer handles
the problem of parameter estimation.
Roughly speaking, model parameters are real numbers that represent properties
of the system that we would like to learn, whereas experiment parameters
represent the choices we get to make in performing measurements.

Model parameters are represented by NumPy arrays of `dtype`_ `float` and that
have two indices, one representing which model is being considered and one
representing which parameter. That is, model parameters are defined by matrices
such that the element :math:`X_{ij}` is the :math:`j^{\text{th}}` parameter of
the model parameter vector :math:`\vec{x}_i`.

By contrast, since not all experiment parameters are best represented by
the data type `float`, we cannot use an array of homogeneous dtype unless there
is only one experimental parameter. The alternative is to use NumPy's
`record array`_ functionality to specify the
*heterogeneous* type of the experiment parameters. To do so, instead of using
a second index to refer to specific experiment parameters, we use *fields*.
Each field then has its own dtype.

For instance, a dtype of ``[('t', 'float'), ('basis', 'int')]`` specifies that
an array has two fields, named ``t`` and ``basis``, having dtypes of ``float``
and ``int``, respectively. Such arrays are initialized by passing lists of
*tuples*, one for each field:

>>> eps = np.array([
...     (12.3, 2),
...     (14.1, 1)
... ], dtype=[('t', 'float'), ('basis', 'int')])
>>> print(eps)
[(12.3, 2) (14.1, 1)]
>>> eps.shape == (2,)
True

Once we have made a record array, we can then index by field names to get out
each field as an array of that field's value in each record, or we can index
by record to get all fields.

>>> print(eps['t'])
[ 12.3  14.1]
>>> print(eps['basis'])
[2 1]
>>> print(eps[0])
(12.3, 2)

Model classes specify the dtypes of their experimental parameters with the
property :attr:`~Simulatable.expparams_dtype`. Thus, a common
idiom is to pass this property to the dtype keyword of NumPy functions. For
example, the model class :class:`~BinomialModel` adds an `int`
field specifying how many times a two-outcome measurement is repeated, so to
specify that we can use its :attr:`~Simulatable.expparams_dtype`:

>>> from qinfer import BinomialModel
>>> bm = BinomialModel(m)
>>> print(bm.expparams_dtype)
[('x', 'float'), ('n_meas', 'uint')]
>>> eps = np.array([
...     (5.0, 10)
... ], dtype=bm.expparams_dtype)


.. _dtype: http://docs.scipy.org/doc/numpy/user/basics.types.html
.. _record array: http://docs.scipy.org/doc/numpy/user/basics.rec.html

Model Outcomes
~~~~~~~~~~~~~~

Given a specific vector of model parameters :math:`\vec{x}` and a specific 
experimental configuration :math:`\vec{c}`, the experiment will 
yield some *outcome* :math:`d` according to the model distribution 
:math:`\Pr(d|\vec{x},\vec{c})`.

In many cases, such as :class:`SimplePrecessionModel` discussed above,
there will be a finite number of outcomes, which we can 
label by some finite set of integers.
For example, we labeled the outcome :math:`\left|0\right\rangle` by
:math:`d=0` and the outcome :math:`\left|1\right\rangle` by 
:math:`d=1`.
If this is the case for you, the rest of this section will likely
not be very relevant, and you may assume your outcomes are 
zero-indexed integers ending at some value.

In other cases, there may be an infinite number of possible outcomes.
For example, if the measurement returns the total number of 
photons measured in a time window, which can in principle 
be arbitrarily large, or if the measuremnt is of a voltage or current, 
which can be any real number.
Or, we may have outcomes which require fancy data types.
For instance, perhaps the output of a single experiment is a tuple 
of numbers rather than a single number.

To accomodate these possible situations, and to have a 
systematic way of testing whether or not all possibe outcomes can be 
enumerated, :class:`Simulatable` (and subclasses like :class:`Model`) 
has a method :attr:`~Simulatable.domain` which for every given 
experimental parameter, returns a :class:`Domain` object.
One major benifit of explicitly storing these objects is that 
certain quantities (like :attr:`~SMCUpdater.bayes_risk`)
can be computed much more efficiently when all possible outcomes 
can be enumerated.
:class:`Domain` has attributes which specify whether or not it 
are finite, how many members it has and what they are, 
what data type they are, and so on.

For the :class:`BinomialModel` defined above, there are 
``n_meas+1`` possible outcomes, with possible values 
the integers between ``0`` and ``n_meas`` inclusive.

>>> bdomain = bm.domain(eps)[0]
>>> bdomain.n_members
11
>>> bdomain.values
array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
>>> bdomain.dtype == np.int
True

We need to extract the :math:`0^\text{th}` element of 
``bm.domain(eps)`` above because ``eps`` is a vector 
of length :math:`1` and :attr:`~Simulatable.domain` always 
returns one domain for every member of ``eps``.
In the case where the domain is completely independent of ``eps``, 
it should be possible to call ``m.domain(None)`` to return 
the unique domain of the model ``m``.

The :class:`MultinomialModel` requires a fancy 
datatype so that outcomes can be tuples of integers.
In the following a single experiment of the model ``mm``
consists of throwing a four sided die ``n_meas`` times 
and recording how many times each side lands facing down.

>>> from qinfer import MultinomialModel, NDieModel
>>> mm = MultinomialModel(NDieModel(n=4))
>>> mm.expparams_dtype
[('exp_num', 'int'), ('n_meas', 'uint')]
>>> mmeps = np.array([(1, 3)], dtype=mm.expparams_dtype)
>>> mmdomain = mm.domain(mmeps)[0]
>>> mmdomain.dtype
dtype([('k', '<i8', (4,))])
>>> mmdomain.n_members
20
>>> print(mmdomain.values)
[([3, 0, 0, 0],) ([2, 1, 0, 0],) ([2, 0, 1, 0],) ([2, 0, 0, 1],)
 ([1, 2, 0, 0],) ([1, 1, 1, 0],) ([1, 1, 0, 1],) ([1, 0, 2, 0],)
 ([1, 0, 1, 1],) ([1, 0, 0, 2],) ([0, 3, 0, 0],) ([0, 2, 1, 0],)
 ([0, 2, 0, 1],) ([0, 1, 2, 0],) ([0, 1, 1, 1],) ([0, 1, 0, 2],)
 ([0, 0, 3, 0],) ([0, 0, 2, 1],) ([0, 0, 1, 2],) ([0, 0, 0, 3],)]

We see here all :math:`20` possible ways to roll this 
die four times.

 .. note::

    :class:`Model` inherits from :class:`Simulatable`, and 
    :class:`FiniteOutcomeModel` inherits from :class:`Model`. 
    The subclass :class:`FiniteOutcomeModel` is able to concretely define 
    some methods (like :attr:`~Simulatable.simulate_experiment`) 
    because of the guarantee that all domains have a finite 
    number of elements.
    Therefore, it is generally a bit less work to construct a 
    :class:`FiniteOutcomeModel` than it is to construct 
    a :class:`Model`.

    Additionally, :class:`FiniteOutcomeModel` automatically 
    defines the domain corresponding to the experimental 
    parameter `ep` by looking at :attr:`~Simulatable.n_outcomes`, 
    namely, if ``nep=n_outcomes(ep)``, then the corresponding 
    domain has members ``0,1,...,nep`` by default.

    Finally, make note of the slightly subtle role of the 
    method :attr:`~Simulatable.n_outcomes`.
    In principle, :attr:`~Simulatable.n_outcomes` is completely 
    independent of :attr:`~Simulatable.domain`. 
    For :class:`FiniteOutcomeModel`, it will almost always hold that 
    ``m.n_outcomes(ep)==domain(ep)[0].n_members``. 
    For models with an infinite number of outcomes, :attr:`~Domain.n_members`
    is not defined, but :attr:`~Simulatable.n_outcomes` is defined
    and refers to "enough outcomes" (at the user's discretion) to 
    make estimates of quantities :attr:`~SMCUpdater.bayes_risk`.

Simulation
~~~~~~~~~~

Both models and simulators allow for simulated data to be drawn from the
model distribution using the :meth:`~Simulatable.simulate_experiment`
method. This method takes a matrix of model parameters and a vector of experiment
parameter records or scalars (depending on the model or simulator),
then returns an array of sample data, one sample for each combination of model
and experiment parameters.

>>> modelparams = np.linspace(0, 1, 100)
>>> expparams = np.arange(1, 10) * np.pi / 2
>>> D = m.simulate_experiment(modelparams, expparams, repeat=3)
>>> print(isinstance(D, np.ndarray))
True
>>> D.shape == (3, 100, 9)
True

If exactly one datum is requested, :meth:`~Simulatable.simulate_experiment`
will return a scalar:

>>> print(m.simulate_experiment(np.array([0.5]), np.array([3.5 * np.pi]), repeat=1).shape)
()

Note that in NumPy, a shape tuple of length zero indicates a scalar value,
as such an array has no indices.

.. note::
    For models with fancy outcome datatypes, it is demanded 
    that the outcome data types, ``[d.dtype for d in m.domain(expparams)]``,
    be identical for every experimental parameter ``expparams`` being 
    simulated. This can be checked with 
    :attr:`~Simulatable.are_expparam_dtypes_consistent`.

.. todo::
    Ensure that the simulated data matches the likelihood.

Likelihooods
~~~~~~~~~~~~

The core functionality of :class:`~Model`, however, is the
:meth:`~Model.likelihood` method. This takes vectors of outcomes,
model parameters and experiment parameters, then returns for each combination
of the three the corresponding probability :math:`\Pr(d | \vec{x}; \vec{e})`.

>>> modelparams = np.linspace(0, 1, 100)
>>> expparams = np.arange(1, 10) * np.pi / 2
>>> outcomes = np.array([0], dtype=int)
>>> L = m.likelihood(outcomes, modelparams, expparams)

The return value of :meth:`~Model.likelihood` is a three-index
array of probabilities whose shape is given by the lengths of ``outcomes``,
``modelparams`` and ``expparams``.
In particular, :meth:`~abstract_model.Model.likelihood` returns a rank-three
tensor :math:`L_{ijk} := \Pr(d_i | \vec{x}_j; \vec{e}_k)`.

>>> print(isinstance(L, np.ndarray))
True
>>> L.shape == (1, 100, 9)
True

Implementing Custom Simulators and Models
-----------------------------------------

In order to implement a custom simulator or model, one must specify metadata
describing the number of outcomes, model parameters, experimental parameters,
etc. in addition to implementing the simulation and/or likelihood methods.

Here, we demonstrate how to do so by walking through a simple subclass of
:class:`~qinfer.abstract_model.FiniteOutcomeModel`. For more detail, please see the
:ref:`apiref`.

Suppose we wish to implement the likelihood function

.. math::

    \Pr(0 | \omega_1, \omega_2; t_1, t_2) = \cos^2(\omega_1 t_1 / 2) \cos^2(\omega_2 t_2 / 2),
    
as may arise in looking, for instance, at an experiment inspired by 2D NMR.
This model has two model parameters, :math:`\omega_1` and :math:`\omega_2`, and
so we start by creating a new class and declaring the number of model
parameters as a `property`:

.. literalinclude:: multicos.py
    :lines: 1-8
    
Next, we proceed to add a property and method indicating that this model always
admits two outcomes, irrespective of what measurement is performed.
This will also automatically define the :attr:`~Simulatable.domain` method.

.. literalinclude:: multicos.py
    :lines: 10-14
    
We indicate the valid range for model parameters by returning an array of
dtype `bool` for each of an input matrix of model parameters, specifying
whether each model vector is valid or not (this is important in resampling,
for instance, to make sure particles don't move to bad locations). Typically,
this will look like some typical bounds checking, combined using
`~numpy.logical_and` and `~numpy.all`. Here, we follow that model by insisting
that *all* elements of each model parameter vector must be at least 0, *and*
must not exceed 1.
    
.. literalinclude:: multicos.py
    :lines: 16-17
    
Next, we specify what a measurement looks like by defining ``expparams_dtype``.
In this case, we want one field that is an array of two `float` elements:

.. literalinclude:: multicos.py
    :lines: 19-21
    
Finally, we write the likelihood itself. Since this is a two-outcome model,
we can calculate the rank-two tensor
:math:`p_{jk} = \Pr(0 | \vec{x}_j; \vec{e}_k)` and let
:meth:`~qinfer.Model.pr0_to_likelihood_array` add an index over
outcomes for us so :math:`L_{0jk}=p_{jk}` and :math:`L_{1jk}=1-p_{jk}`.
To compute :math:`p_{jk}` efficiently, it is helpful to do a bit of index
gymnastics  using NumPy's powerful `broadcasting rules`_. In this example, we
set up the calculation to produce terms of the form
:math:`\cos^2(x_{j,l} e_{k,l} / 2)` for :math:`l \in \{0, 1\}` indicating
whether we're referring to :math:`\omega_1` or :math:`\omega_2`, respectively.
Multiplying along this axis then gives us the product of the two cosine
functions, and in a way that very nicely generalizes to likelihood functions of
the form

.. math::

    \Pr(0 | \omega_1, \omega_2; t_1, t_2) = \prod_l \cos^2(\omega_l t_l / 2).
    
Running through the index gymnastics, we can implement the likelihood function
as:

.. literalinclude:: multicos.py
    :lines: 23-48
    :emphasize-lines: 35-43
    
Our new custom model is now ready to use! To simulate data from this model, we
set up ``modelparams`` and ``expparams`` as before, taking care to conform to
the ``expparams_dtype`` of our model:

.. testsetup::

    import os, sys
    sys.path.insert(0, os.path.join(os.getcwd(), 'source', 'guide'))
    from multicos import MultiCosModel

>>> mcm = MultiCosModel()
>>> modelparams = np.dstack(np.mgrid[0:1:100j,0:1:100j]).reshape(-1, 2)
>>> expparams = np.empty((81,), dtype=mcm.expparams_dtype)
>>> expparams['ts'] = np.dstack(np.mgrid[1:10,1:10] * np.pi / 2).reshape(-1, 2)
>>> D = mcm.simulate_experiment(modelparams, expparams, repeat=2)
>>> print(isinstance(D, np.ndarray))
True
>>> D.shape == (2, 10000, 81)
True

.. note::

    Creating ``expparams`` as an empty array and filling it by field name is a
    straightforward way to make sure it matches ``expparams_dtype``, but it
    comes with the risk of forgetting to initialize a field, so take care when
    using this method.

.. _broadcasting rules: http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

.. _models_guide_derived:

Adding Functionality to Models with Other Models
------------------------------------------------

QInfer also provides model classes which add functionality or otherwise modify
other models. For instance, the :class:`BinomialModel` class accepts instances
of two-outcome models and then represents the likelihood for many repeated
measurements of that model. This is especially useful in cases where
experimental concerns make switching experiments costly, such that repeated
measurements make sense.

To use :class:`BinomialModel`, simply provide an instance of another model
class:

>>> from qinfer import SimplePrecessionModel
>>> from qinfer import BinomialModel
>>> bin_model = BinomialModel(SimplePrecessionModel())

Experiments for :class:`BinomialModel` have an additional field from the
underlying models, called ``n_meas``. If the original model used scalar
experiment parameters (e.g.: ``expparams_dtype`` is `float`), then the original
scalar will be referred to by a field ``x``.

>>> eps = np.array([(12.1, 10)], dtype=bin_model.expparams_dtype)
>>> print(eps['x'], eps['n_meas'])
[ 12.1] [10]

Another model which *decorates* other models in this way is :class:`PoisonedModel`,
which is discussed in more detail in :ref:`perf_testing_guide`. Roughly,
this model causes the likeihood functions calculated by its underlying model
to be subject to random noise, so that the robustness of an inference algorithm
against such noise can be tested.

