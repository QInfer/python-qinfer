=================
Welcome to QInfer
=================

.. image:: https://zenodo.org/badge/19664/QInfer/python-qinfer.svg
   :target: https://zenodo.org/badge/latestdoi/19664/QInfer/python-qinfer

.. image:: https://travis-ci.org/QInfer/python-qinfer.svg?branch=master
    :target: https://travis-ci.org/QInfer/python-qinfer

.. image:: https://coveralls.io/repos/github/QInfer/python-qinfer/badge.svg?branch=master
    :target: https://coveralls.io/github/QInfer/python-qinfer?branch=master 

.. image:: https://codeclimate.com/github/QInfer/python-qinfer/badges/gpa.svg
   :target: https://codeclimate.com/github/QInfer/python-qinfer
   :alt: Code Climate

**QInfer** is a library using Bayesian sequential Monte Carlo for quantum
parameter estimation.


Obtaining QInfer
================

A stable version of **QInfer** has not yet been released. Until then,
the latest version may always be obtained by cloning into the GitHub
repository::

    $ git clone git@github.com:csferrie/python-qinfer.git
    
Once obtained in this way, **QInfer** may be updated by pulling from GitHub::

    $ git pull

Installing QInfer
=================

**QInfer** provides a setup script, ``setup.py``, for installing from source.
On Unix-like systems, **QInfer** can be made available globally by running::

    $ cd /path/to/qinfer/
    $ sudo python setup.py install

On Windows, run ``cmd.exe``, then run the setup script::

    C:\> cd C:\path\to\qinfer\
    C:\path\to\qinfer\> python setup.py install
    
Note that you may be prompted for permission by User Access Control.

Dependencies
============

**QInfer** depends on only a very few packages:

- Python 2.7 (may work with earlier, but not tested).
- NumPy and SciPy.
- [Optional] `SciKit-Learn`_ required for some advanced features.
- [Optional] `Sphinx`_ required to rebuild documentation.

On Windows, these packages can be provided by `Python(x,y)`_. Linux users may
obtain the needed dependencies. Under Ubuntu::

    $ sudo apt-get install python2.7 python-numpy python-scipy python-scikits-learn python-sphinx
    
On Fedora::

    $ sudo yum install python numpy scipy python-sphinx
    $ sudo easy_install -U scikit-learn

Alternatively,
`Enthought Python Distribution`_ has been tested with **QInfer**, and may be
used on Windows, Mac OS X or Linux.

More Information
================

Full documentation for **QInfer** is
`available on ReadTheDocs <http://python-qinfer.readthedocs.org/en/latest/>`_,
or may be built locally by running the documentation
build script in ``doc/``::

    $ cd /path/to/qinfer/doc/
    $ make html
    
On Windows::
    
    C:\> cd C:\path\to\qinfer\
    C:\path\to\qinfer\> make.bat html
    
The generated documentation can be viewed by opening
``doc/_build/html/index.html``.

.. _Enthought Python Distribution: http://www.enthought.com/products/epd.php
.. _Python(x,y): http://code.google.com/p/pythonxy/
.. _SciKit-Learn: http://scikit-learn.org/stable/
.. _Sphinx: http://sphinx-doc.org/
