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
:class:`~abstract_model.Model`, when the likelihood function can be numerically
evaluated, or :class:`~abstract_model.Simulatable` when only samples from the
function can be efficiently generated.

Using Models and Simulations
----------------------------

Basic Functionality
~~~~~~~~~~~~~~~~~~~

Both :class:`~abstract_model.Model` and :class:`~abstract_model.Simulatable` offer
basic functionality to describe how they are parameterized, what outcomes are
possible, etc. For this example, we will use a premade model from :mod:`test_models`,
:class:`~test_models.SimplePrecessionModel`. This model implements the likelihood
function

.. math::

    \Pr(d | \omega; t) = \begin{cases}
        \cos^2 (\omega t / 2) & d = 0 \\
        \sin^2 (\omega t / 2) & d = 1
    \end{cases},

as can be derived from Born's Rule for a spin-½ particle prepared and measured
in the :math:`\left|+\right\rangle` state, and evolved under
:math:`H = \omega \sigma_z / 2` for some time :math:`t`.

In this way, we see that by defining the likelihood function in terms of the
hypothetical outcome :math:`d`, the model parameter :math:`\omega`, and the experimental
parameter :math:`t`, we can reason about the experimental data that we would extract
from the system.

In order to use this likelihood function, we must instantiate the model that
implements the likelihood. Since :class:`~test_models.SimplePrecessionModel` is
provided with QInfer, we can simply import it and make an instance.

>>> from qinfer.test_models import SimplePrecessionModel
>>> m = SimplePrecessionModel()

Once a model or simulator has been created, you can query how many model
parameters it admits and how many outcomes a given experiment can have.

>>> print m.n_modelparams
1
>>> print m.modelparam_names
['\\omega']
>>> print m.is_n_outcomes_constant
True
>>> print m.n_outcomes(expparams=0)
2

Model and Experiment Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The division between unknown parameters that we are trying to learn (:math:`\omega`
in the ``SimplePrecessionModel`` example) and the controls that we can use to
design measurements (:math:`t`) is generic, and is key to how QInfer handles
the problem of parameter estimation.
Roughly speaking, model parameters are real numbers that represent properties
of the system that we would like to learn, whereas experiment parameters
represent the choices we get to make in performing measurements.

Model parameters are represented by NumPy arrays of `dtype`_ `float` and that
have two indices, one representing which model is being considered and one
representing which parameter. That is, model parameters are defined by matrices
such that the element :math:`X_ij` is the :math:`j^{\text{th}}` parameter of
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

>>> import numpy as np
>>> eps = np.array([
...     (12.3, 2),
...     (14.1, 1)
... ], dtype=[('t', 'float'), ('basis', 'int')])
>>> print eps
[(12.3, 2) (14.1, 1)]
>>> print eps.shape
(2,)

Once we have made a record array, we can then index by field names to get out
each field as an array of that field's value in each record, or we can index
by record to get all fields.

>>> print eps['t']
[ 12.3  14.1]
>>> print eps['basis']
[2 1]
>>> print eps[0]
(12.3, 2)

Model classes specify the dtypes of their experimental parameters with the
property :attr:`~abstract_model.Simulatable.expparams_dtype`. Thus, a common
idiom is to pass this property to the dtype keyword of NumPy functions. For
example, the model class :class:`~derived_models.BinomialModel` adds an `int`
field specifying how many times a two-outcome measurement is repeated, so to
specify that we can use its :attr:`~abstract_model.Simulatable.expparams_dtype`:

>>> from qinfer.derived_models import BinomialModel
>>> bm = BinomialModel(m)
>>> print bm.expparams_dtype
[('x', 'float'), ('n_meas', 'uint')]
>>> eps = np.array([
...     (11.0, 20)
... ], dtype=bm.expparams_dtype)


.. _dtype: http://docs.scipy.org/doc/numpy/user/basics.types.html
.. _record array: http://docs.scipy.org/doc/numpy/user/basics.rec.html

Simulation
~~~~~~~~~~

Both models and simulators allow for simulated data to be drawn from the
model distribution using the :meth:`~abstract_model.Simulatable.simulate_experiment`
method. This method takes a matrix of model parameters and a vector of experiment
parameter records or scalars (depending on the model or simulator),
then returns an array of sample data, one sample for each combination of model
and experiment parameters.

>>> import numpy as np
>>> modelparams = np.linspace(0, 1, 100)
>>> expparams = np.arange(1, 10) * np.pi / 2
>>> D = m.simulate_experiment(modelparams, expparams, repeat=3)
>>> print type(D)
<type 'numpy.ndarray'>
>>> print D.shape
(3, 100, 9)

If exactly one datum is requested, :meth:`~abstract_model.Simulatable.simulate_experiment`
will return a scalar:

>>> print m.simulate_experiment(np.array([0.5]), np.array([3.5 * np.pi]), repeat=1).shape
()

Note that in NumPy, a shape tuple of length zero indicates a scalar value,
as such an array has no indices.

.. todo::
    Ensure that the simulated data matches the likelihood.

Likelihooods
~~~~~~~~~~~~

The core functionality of :class:`~abstract_model.Model`, however, is the
:meth:`~abstract_model.Model.likelihood` method. This takes vectors of outcomes,
model parameters and experiment parameters, then returns for each combination
of the three the corresponding probability :math:`\Pr(d | \vec{x}; \vec{e})`.

>>> import numpy as np
>>> modelparams = np.linspace(0, 1, 100)
>>> expparams = np.arange(1, 10) * np.pi / 2
>>> outcomes = np.array([0], dtype=int)
>>> L = m.likelihood(outcomes, modelparams, expparams)

The return value of :meth:`~abstract_model.Model.likelihood` is a three-index
array of probabilities whose shape is given by the lengths of ``outcomes``,
``modelparams`` and ``expparams``.
In particular, :meth:`~abstract_model.Model.likelihood` returns a rank-three
tensor :math:`L_{ijk} := \Pr(d_i | \vec{x}_j; \vec{e}_k)`.

>>> print type(L)
<type 'numpy.ndarray'>
>>> print L.shape
(1, 100, 9)

Implementing Custom Models
--------------------------

TODO
