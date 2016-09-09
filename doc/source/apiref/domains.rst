..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. currentmodule:: qinfer

Domains
=======

Introduction
------------

A :class:`Domain` represents a collection of objects. They 
are used by :class:`Simulatable` (and subclasses like 
:class:`Model` and :class:`FiniteOutcomeModel`) to store 
relevant information about the possible outcomes of a given 
experiment.
This includes properties like whether or not there are 
a finite number of possibilities, if so how many, and 
what their data types are.

:class:`Domain` - Base Class for Domains
----------------------------------------

All domains should inherit from this base class.

Class Reference 
~~~~~~~~~~~~~~~
.. autoclass:: Domain
    :members:
    
:class:`RealDomain` - (A subset of) Real Numbers
------------------------------------------------

Class Reference 
~~~~~~~~~~~~~~~
.. autoclass:: RealDomain
    :members:

:class:`IntegerDomain` - (A subset of) Integers
-----------------------------------------------

This is the default domain for :class:`FiniteOutcomeModel`.

Class Reference 
~~~~~~~~~~~~~~~
.. autoclass:: IntegerDomain
    :members:

:class:`MultinomialDomain` - Tuples of Integers with a Constant Sum
-------------------------------------------------------------------

This domain is used by :class:`MultinomialModel`.

Class Reference 
~~~~~~~~~~~~~~~
.. autoclass:: MultinomialDomain
    :members: