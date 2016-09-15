..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _derived_models:
    
.. currentmodule:: qinfer

Derived Models
==============

Introduction
------------

QInfer provides several models which *decorate* other models, providing
additional functionality or changing the behaviors of underlying models.

:class:`PoisonedModel` - Model corrupted by likelihood errors
-------------------------------------------------------------

.. autoclass:: PoisonedModel
    :members:
    
:class:`BinomialModel` - Model over batches of two-outcome experiments
----------------------------------------------------------------------

.. autoclass:: BinomialModel
    :members:

:class:`MultinomialModel` - Model over batches of D-outcome experiments
----------------------------------------------------------------------

.. autoclass:: MultinomialModel
    :members:

:class:`MLEModel` - Model for approximating maximum-likelihood estimation
-------------------------------------------------------------------------

.. autoclass:: MLEModel
    :members:
