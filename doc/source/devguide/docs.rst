..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _docs_devguide:
    
.. currentmodule:: qinfer

Writing Documentation
=====================

The documentation for **QInfer** is written using the `Sphinx
documentation engine`_, and is hosted by `ReadTheDocs`_.
This allows us to produce high-quality HTML- and LaTeX-formatted
reference and tutorial material for **QInfer**. In particular, the
documentation `hosted by ReadTheDocs`_ is integrated with the
`GitHub project`_ for **QInfer**, simplifying the process of building
and deploying documentation.

In this section, we will discuss how to use Sphinx and ReadTheDocs
together to contribute to and improve **QInfer** documentation. 

.. _Sphinx documentation engine: http://www.sphinx-doc.org/en/stable/
.. _ReadTheDocs: readthedocs.org
.. _hosted by ReadTheDocs: http://python-qinfer.readthedocs.io/
.. _GitHub project: https://github.com/QInfer/python-qinfer

Building Documentation With Sphinx
----------------------------------

In developing and writing documentation, it is helpful
to be able to compile the current version of the documentation
offline. To do so, first install Sphinx itself. If you are using
Anaconda:

.. code:: bash

    $ conda install sphinx

Otherwise, we suggest installing Sphinx using `pip`_:

.. code:: bash

    $ pip install sphinx

In either case, after installing Sphinx, you may need to install
additional libraries that used by particular examples in the **QInfer**
documentation. On Anaconda:

.. code:: bash

    $ conda install scikit-learn ipython future matplotlib
    $ conda install -c conda-forge qutip
    $ pip install mpltools

Otherwise:

.. code:: bash

    $ pip install -r doc/rtd-requirements.txt

With the dependencies installed, you can now build the documentation
using the ``Makefile`` provided by Sphinx, or using the ``make.bat``
script for Windows. Note that because the documentation includes several
computationally-intensive examples, the build process may take a significant
amount of time (a few minutes). On Linux and OS X:

.. code:: bash

    $ cd doc/
    $ make clean # Deletes all previously compiled outputs.
    $ make html # Builds HTML-formatted docs.
    $ make latexpdf # Builds PDF-formatted docs using LaTeX.

All of the compiled outputs will be saved to the ``doc/_build``
folder. In particular, the HTML version can be found at
``doc/_build/html/index.html``.

On Windows, we recommend using PowerShell to run ``make.bat``:

.. code:: powershell

    PS > cd doc/
    PS > .\make.bat clean
    PS > .\make.bat html

Note that on Windows, building PDF-formatted docs requires an additional
step. First, make the LaTeX-formatted source:

.. code:: powershell

    PS > .\make latex

This will produce a folder called ``doc/_build/latex`` containing
``QInfer.tex``. Build this with your favorite LaTeX front-end to
produce the final PDF.

.. _pip: https://pip.pypa.io/en/stable/

Formatting Documentation With reStructuredText
----------------------------------------------

The documentation itself is written in the reStructuredText language,
an extensible and (largely) human-readable text format. We recommend
reading the `primer`_ provided with Sphinx to get a start. Largely, however,
documentation can be written as plain text, with emphasis indicated
by ``*asterisks*``, strong text indicated by ``**double asterisks**``,
and verbatim snippets indicated by ````double backticks````.
Sections are denoted by different kinds of underlining,
using ``=``, ``-``, ``~`` and ``^`` to indicate sections, subsections,
paragraphs and subparagraphs, respectively.

Links are a bit more complicated, and take on a couple several
different forms:

- Inline links consist of backticks, with addresses denoted in angle-brackets,
  ```link text <link target>`_``. Note the final ``_``, which denotes that
  the backticks describe a link.

- Alternatively, the link target may be placed later on the page, as in the
  following snippet::

      Lorem `ipsum`_ dolor sit amet...

      .. _ipsum: http://www.lipsum.com/

- Links within the documentation are made using ``:ref:``. For example,
  ``:ref:`apiref`:`` formats as :ref:`apiref`. The target of such a reference
  must be declared before a section header, as in the following example, which
  declares the target ``foobar``::

      .. _foobar:

      A Foo About Bar
      ---------------

- Links to Python classes, modules and functions are formatted using
  ``:class:``, ``:mod:`` and ``:func:``, repsectively. For example,
  ``:class:`qinfer.SMCUpdater`` formats as :class:`qinfer.SMCUpdater`.
  To suppress the path to a Python name, preface the name with a
  tilde (``~``), as in ``:class:`~qinfer.SMCUpdater:```. For a Python
  name to be a valid link target, it must be listed in the :ref:`apiref`
  (see below), or must be documented in one of the external projects
  listed in ``doc/conf.py``. For instance, to link to NumPy documentation,
  use a link of the form ``:class:`~numpy.ndarray```.

- Finally, **QInfer** provides special notation for linking to DOIs,
  Handles and arXiv postings::

      :doi:`10.1088/1367-2630/18/3/033024`, :hdl:`10012/9217`,
      :arxiv:`1304.5828`

.. _primer: http://www.sphinx-doc.org/en/stable/rest.html

Typesetting Math
~~~~~~~~~~~~~~~~

Math is formatted using the Sphinx markup ``:math:`...``` in place
of ``$``, and using the ``.. math::`` directive in place of ``$$``.
When building HTML- or PDF-formatted documentation, this is automatically
converted to `MathJax`_- or LaTeX-formatted math. The **QInfer**
documentation is configured with several macros available to each of
MathJax and LaTeX, specified in ``doc/_templates/page.html``
and ``doc/conf.py``, respectively. For example, ``:math:`\expect```
is configured to produce the blackboard-bold expectation operator
:math:`\expect`.

.. _MathJax: https://www.mathjax.org/

Docstrings and API References
-----------------------------

One of the most useful features of Sphinx is that it can import documentation
from Python code itself. In particular, the ``.. autofunction::`` and
``.. autoclass::`` directives import documentation from functions and classes,
respectively. These directives typeset the docstrings for their targets as
reStructuredText, with the following notation used to indicate arguments,
return types, etc.:

- ``:param name:`` is used to declare that a class' initializer or a function
  takes an argument named ``name``, and is followed by a description of that
  parameter.
- ``:param type name:`` can be used to indicate that ``name`` has the type
  type ``type``. If ``type`` is a recognized Python type, then Sphinx will
  automatically convert it into a link.
- ``:type name:`` can be used to provide more detailed information about
  the type of the parameter ``name``, and is followed by a longer description
  of that parameter's type. ``:type:`` on its own can be used to denote
  the type of a property accessor.
- ``:return:`` is used to describe what a function returns.
- ``:rtype:`` is used to describe the type of a return value.
- ``:raises exc_type:`` denotes that the described function raises an
  exception of type ``exc_type``, and describes the conditions under which
  that exception is raised.

In addition to the standard Sphinx fields described above, **QInfer**
adds the following fields:

- ``:modelparam name:`` describes a model parameter named ``name``,
  where the name is formatted as math (or should be, pending bugs
  in the documentation configuration).
- ``:expparam field_type field_name:`` describes an experiment parameter
  field named ``field_name`` with :class:`~numpy.dtype` ``field_type``.
- ``:scalar-expparam scalar_type`` describes that a class has exactly
  one experiment parameter of type ``scalar_type``.
- ``:column dtype name:`` describes a column for data taken by
  :ref:`simple_est_guide` functions.

Importantly, if math is included in a docstring, it is highly
recommended to format the docstring as a *raw string*; that is,
as a string starting with ``r'`` or ``r"`` for inline strings
or ``r'''`` or ``r"""`` for multi-line strings. This avoids having
to escape TeX markup that appears within a docstring. For instance,
consider the following hypothetical function:

.. code-block:: python

    def state(theta, phi):
        r"""
        Returns an array representing :math:`\cos(\theta)\ket{0} +
        \sin(\theta) e^{i \phi} \ket{1}`. 
        """
        ...

If the docstring were instead declared using ``"""``, then
``\t`` everywhere inside the docstring would be interpreted by
Python as a tab character, and not as the start of a LaTeX command.  

.. _docs_devguide_snippets:

Showing Code Snippets
---------------------

The documentation would be useless without code snippets, so Sphinx
provides several ways to show snippets. Perhaps the most common is
*doctest-style*::

    Lorem ipsum dolor sit amet...

    >>> print("Hello, world!")
    Hello, world!

As described in :ref:`_doctest_devguide`, these snippets are run
automatically as *tests* to ensure that the documentation and code
are both correct. To mark a particular line in a snippet as not
testable, add ``# doctest: +SKIP`` as a comment after.

For longer snippets, or for snippets that should not be run as
tests, use the ``.. code:: python`` directive::

    .. code:: python

        print("Hello, world!")

This formats as:

.. code:: python

    print("Hello, world!")

Finally, a block can be formatted as code without any syntax highlighting
by using the ``::`` notation on the previous line, and then indenting
the block itself::

    This is in fact how this section denotes reStructuredText code samples::

        :foobar: is not a valid reStructuredText role.

Plotting Support
----------------

Sphinx can also run Python code and plot the resulting figures.
To do so, use the ``.. plot::`` directive. Note that any such plots
should be relatively quick to generate, especially so as to not
overburden the build servers provided by `ReadTheDocs`_.
The ``.. plot::`` directive has been configured to automatically
import **QInfer** itself and to import `matplotlib`_ as ``plt``.
Thus, for example, the following demonstrates plotting functionality
provided by the ``qinfer.tomography`` module::

    .. plot::

        basis = tomography.bases.pauli_basis(1)
        prior = tomography.distributions.GinibreDistribution(basis)
        tomography.plot_rebit_prior(prior, rebit_axes=[1, 3])
        plt.show()

Note the ``plt.show()`` call at the end; this is *required* to produce
the final figure!

.. _matplotlib: http://matplotlib.org/