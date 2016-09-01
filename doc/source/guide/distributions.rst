..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _distributions_guide:
    
.. currentmodule:: qinfer

Representing Probability Distributions
======================================

Introduction
------------

Probability distributions such as prior distributions over model parameters
are reprented in QInfer by objects of type :class:`Distribution` that are
responsible for producing samples according to those distributions. This is
especially useful, for instance, when drawing initial particles for use with
an :class:`~qinfer.smc.SMCUpdater`.

The approach to representing distributions taken by QInfer is somewhat
different to that taken by, for example, :mod:`scipy.stats`, in that
a QInfer :class:`Distribution` is a class that produces samples according to
that distribution. This means that QInfer :class:`Distribution` objects provide much
less information than do those represented by objects in :mod:`scipy.stats`,
but that they are much easier to write and combine.

Sampling Pre-made Distributions
-------------------------------

QInfer comes along with several distributions, listed in
:ref:`specific_distributions`. Each of these is a subclass of
:class:`Distribution`, and hence has a method :meth:`~Distribution.sample`
that produces an array of samples.

>>> from qinfer import NormalDistribution
>>> dist = NormalDistribution(0, 1)
>>> samples = dist.sample(n=5)
>>> samples.shape == (5, 1)
True

Combining Distributions
-----------------------

Distribution objects can be combined using other distribution objects. For
instance, if :math:`a \sim \mathcal{N}(0, 1)` and :math:`b \sim \text{Uni}(0, 1)`,
then the product distribution on :math:`(a,b)` can be produced by using
:class:`ProductDistribution`:

>>> from qinfer import UniformDistribution, ProductDistribution
>>> a = NormalDistribution(0, 1)
>>> b = UniformDistribution([0, 1])
>>> ab = ProductDistribution(a, b)
>>> samples = ab.sample(n=5)
>>> samples.shape == (5, 2)
True

    
Making Custom Distributions
---------------------------

To make a custom distribution, one need only implement
:meth:`~Distribution.sample` and set the property :attr:`~Distribution.n_rvs`
to indicate how many random variables the new distribution class represents.

For example, to implement a distribution over :math:`x` and :math:`y` such that
:math:`\sqrt{x^2 + y^2} \sim \mathcal{N}(1, 0.1)` and such that
the angle between :math:`x` and :math:`y` is drawn from
:math:`\text{Uni}(0, 2 \pi)`:

.. code-block:: python

    from qinfer import Distribution
    
    class RingDistribution(Distribution):
        @property
        def n_rvs(self):
            return 2
            
        def sample(self, n=1):
            r = np.random.randn(n, 1) * 0.1 + 1
            th = np.random.random((n, 1)) * 2 * np.pi
            
            x = r * np.cos(th)
            y = r * np.sin(th)
            
            return np.concatenate([x, y], axis=1)

