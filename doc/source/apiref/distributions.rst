..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.

.. _distributions:

.. currentmodule:: qinfer

Probability Distributions
=========================

.. seealso::

    :ref:`Specific distributions (tomography) <tomography_distributions>`

:class:`Distribution` - Abstract Base Class for Probability Distributions
-------------------------------------------------------------------------

.. autoclass:: Distribution
    :members:

.. _specific_distributions:

Specific Distributions
----------------------

.. autoclass:: UniformDistribution
    :members:

.. autoclass:: DiscreteUniformDistribution
    :members:

.. autoclass:: MVUniformDistribution
    :members:

.. autoclass:: NormalDistribution
    :members:

.. autoclass:: MultivariateNormalDistribution
    :members:

.. autoclass:: SlantedNormalDistribution
    :members:

.. autoclass:: LogNormalDistribution
    :members:

.. autoclass:: ConstantDistribution
    :members:

.. autoclass:: BetaDistribution
    :members:

.. autoclass:: BetaBinomialDistribution
    :members:

.. autoclass:: GammaDistribution
    :members:

.. autoclass:: InterpolatedUnivariateDistribution
    :members:

.. autoclass:: HilbertSchmidtUniform
    :members:

.. autoclass:: HaarUniform
    :members:

.. autoclass:: GinibreUniform
    :members:

.. autoclass:: ParticleDistribution
    :members:

Combining Distributions
-----------------------

QInfer also offers classes for combining distributions together to produce new
ones.

.. autoclass:: ProductDistribution
    :members:

.. autoclass:: PostselectedDistribution
    :members:

.. autoclass:: MixtureDistribution
    :members:

.. autoclass:: ConstrainedSumDistribution
    :members:

Mixins for Distribution Development
-----------------------------------

.. autoclass:: SingleSampleMixin
    :members:
