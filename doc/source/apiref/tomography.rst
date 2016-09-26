..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _tomography:
    
.. currentmodule:: qinfer.tomography

Quantum Tomography
==================


:class:`TomographyBasis`
------------------------

.. autoclass:: TomographyBasis
    :members:

.. _tomography_bases:

Built-in bases
~~~~~~~~~~~~~~

.. autofunction:: gell_mann_basis

.. autofunction:: pauli_basis

.. autofunction:: tensor_product_basis


:class:`DensityOperatorDistribution`
------------------------------------

.. autoclass:: DensityOperatorDistribution
    :members:

.. _tomography_distributions:

Specific Distributions
----------------------

.. seealso::

    :ref:`distributions`

.. autoclass:: TensorProductDistribution
    :members:

.. autoclass:: GinibreDistribution
    :members:

.. autoclass:: GinibreReditDistribution
    :members:

.. autoclass:: BCSZChoiDistribution
    :members:

.. autoclass:: GADFLIDistribution
    :members:

Models
------

.. autoclass:: TomographyModel
    :members:

.. autoclass:: DiffusiveTomographyModel
    :members:

Heuristics
----------

.. _tomography_heuristics:

Abstract Heuristics
~~~~~~~~~~~~~~~~~~~

.. autoclass:: StateTomographyHeuristic
    :members:

.. autoclass:: ProcessTomographyHeuristic
    :members:

.. autoclass:: BestOfKMetaheuristic
    :members:

Specific Heuristics
~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomStabilizerStateHeuristic
    :members:

.. autoclass:: RandomPauliHeuristic
    :members:

.. autoclass:: ProductHeuristic
    :members:

Plotting Functions
------------------

.. autofunction:: plot_cov_ellipse

.. autofunction:: plot_rebit_prior

.. autofunction:: plot_rebit_posterior