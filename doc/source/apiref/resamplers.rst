..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _resamplers:
    
.. currentmodule:: qinfer

Resampling Algorithms
=====================

Introduction
------------

In order to restore numerical stability to the sequential Monte Carlo
algorithm as the effective sample size is reduced, *resampling* is used to
adaptively move particles so as to better represent the posterior distribution.
**QInfer** allows for such algorithms to be specified in a modular way.

:class:`Resampler` - Abstract base class for resampling algorithms
------------------------------------------------------------------

Class Reference
~~~~~~~~~~~~~~~

.. autoclass:: Resampler
    :members:
    :special-members: __call__

:class:`LiuWestResampler` -  Liu and West (2000) resampling algorithm
---------------------------------------------------------------------

Class Reference
~~~~~~~~~~~~~~~

.. autoclass:: LiuWestResampler
    :members:
    :special-members: __call__
