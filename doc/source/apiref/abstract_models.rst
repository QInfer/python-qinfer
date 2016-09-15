..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. currentmodule:: qinfer

Abstract Model Classes
======================

Introduction
------------

A *model* in QInfer is a class that describes the probabilities of observing
data, given a particular experiment and given a particular set of model
parameters. The observation probabilities may be given implicitly or explicitly,
in that the class may only allow for sampling observations, rather than finding
the a distribution explicitly. In the former case, a model is represented by
a subclass of :class:`Simulatable`, while in the latter, the model is
represented by a subclass of :class:`Model`.

:class:`Simulatable` - Base Class for Implicit (Simulatable) Models
-------------------------------------------------------------------

Class Reference 
~~~~~~~~~~~~~~~
.. autoclass:: Simulatable
    :members:
    
:class:`Model` - Base Class for Explicit (Likelihood) Models
------------------------------------------------------------
    
If a model supports explicit calculation of the likelihood function, then this
is represented by subclassing from :class:`Model`. 
    
Class Reference
~~~~~~~~~~~~~~~
    
.. autoclass:: Model
    :members:

:class:`FiniteOutcomeModel` - Base Class for  Models with a Finite Number of Outcomes
-------------------------------------------------------------------------------------
    
The likelihood function provided by a subclass is used to implement
:meth:`Simulatable.simulate_experiment`, which is possible because the 
likelihood of all possible outcomes can be computed.
This class also concretely implements the :attr:`~Simulatable.domain` method 
by looking at the definition of :attr:`~Simulatable.n_outcomes`.
    
Class Reference
~~~~~~~~~~~~~~~
    
.. autoclass:: FiniteOutcomeModel
    :members:
        
:class:`DifferentiableModel` - Base Class for Explicit Models with Differentiable Likelihoods
---------------------------------------------------------------------------------------------

Class Reference
~~~~~~~~~~~~~~~
    
.. autoclass:: DifferentiableModel
    :members:
