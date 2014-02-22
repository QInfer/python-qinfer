..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _distributions:
    
.. currentmodule:: qinfer.distributions

Probability Distributions
=========================

:class:`Distribution` - Abstract Base Class for Probability Distributions
-------------------------------------------------------------------------

.. autoclass:: Distribution
    :members:

.. _specific_distributions:

Specific Distributions
----------------------

.. autoclass:: UniformDistribution
    :members:

.. autoclass:: ConstantDistribution
    :members:

.. autoclass:: HilbertSchmidtUniform
    :members:
    
.. autoclass:: HaarUniform
    :members:
    
.. autoclass:: GinibreUniform
    :members:
    
Combining Distributions
-----------------------

QInfer also offers classes for combining distributions together to produce new
ones.

.. autoclass:: ProductDistribution
    :members:

.. autoclass:: PostselectedDistribution
    :members

