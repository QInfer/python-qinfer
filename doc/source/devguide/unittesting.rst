..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _unittesting_devguide:
    
.. currentmodule:: qinfer

Unit and Documentation Testing
==============================

Unit Tests
----------

.. _doctest_devguide:

Documentation Tests
-------------------

As described in :ref:`docs_devguide_snippets`, Sphinx integrates with
Python's :mod:`doctest` module to help ensure that both the documentation
and underlying library are correct. Doctests consist of short snippets
of code along with their expected output. A doctest *passes* if the
actual output of the snippet matches its expected output. For instance,
a doctest that ``1 + 1`` correctly produces ``2`` could be written as::

    Below, we show an example of addition in practice:

    >>> 1 + 1
    2

    Later, we will consider more advanced operators.

The blank lines above and below the doctest separate it from the surrounding
text, while the expected output appears immediately below the relevant code.

To run the doctests in the **QInfer** documentation using Linux or OS X:

.. code-block:: bash

    $ cd doc/
    $ make doctest

To run the doctests on Windows using PowerShell, use ``.\make`` instead:

.. code-block:: bash

    PS > cd doc/
    PS > .\make doctest

As with the unit tests, doctests are automatically run on pull requests,
to help ensure the correctness of contributed documentation.

Test Annotations
~~~~~~~~~~~~~~~~

A doctest snippet may be annotated with one or more comments that change
the behavior of that test. The :mod:`doctest` documentation goes into far
more detail, but the two we will commonly need are ``# doctest: +SKIP``
and ``# doctest: +ELLIPSIS``. The former causes a test to be skipped entirely.
Skipping tests can be useful if the output of a doctest is random, for instance.

The second, ``+ELLIPSIS``, causes any ellipsis (``...``) in the expected output
to act as a wild card. For instance, both of the following doctests would pass::

    >>> print([1, 2, 3]) # doctest: +ELLIPSIS
    [1, ..., 3]
    >>> print([1, 2, 3, 4, 'foo', 3]) # doctest: +ELLIPSIS
    [1, ..., 3]

Doctest Annoyances
~~~~~~~~~~~~~~~~~~

There are a few annoyances that come along with writing tests based
on string-equivalence of outputs, in particular for a cross-platform
and 2/3 compatible library. In particular:

- NumPy, 2to3 and ``type``/``class``: Python 2 and 3 differ on whether
  the type of NumPy :class:`~numpy.ndarray` prints as ``type`` (Python 2)
  ``class`` (Python 3)::

      >>> print(type(np.array([1])))
      <type 'numpy.ndarray'> # Python 2
      <class 'numpy.ndarray'> # Python 3

  Thus, to write a doctest that checks if something is a NumPy array or not,
  it is preferred to use an `isinstance` check instead::

      >>> isinstance(np.array([1]), np.ndarray)
      True

  Though this sacrifices some on readability, it gains on portability and correctness.

- `int` versus `long` in array shapes: The Windows and Linux versions of NumPy
  behave differently with respect to when a NumPy shape is represented as a
  `tuple` of `int` versus a `tuple` of `long` values. This can cause doctests
  to choke on spurious ``L`` suffixes::

      >>> print(np.zeros((1, 10)).shape)
      (1, 10) # tuple of int
      (1L, 10L) # tuple of long

  Since `int` and `long` respect ``==``, however, the same trick as above can
  help::

      >>> np.zeros((1, 10)).shape == (1, 10)
      True

