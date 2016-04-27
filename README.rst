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
parameter estimation. Works with Python 2.7, 3.3 and 3.4.

Installing QInfer
=================

We recommend using **QInfer** with the
`Anaconda distribution`_. Download and install
Anaconda for your platform, either Python 2 or 3. We
suggest using Python 3, but **QInfer**
works with either. Once Anaconda is installed, simply run ``pip`` to install **QInfer**::

    $ pip install git+https://github.com/QInfer/python-qinfer.git

Alternatively, **QInfer** can be installed manually by downloading from GitHub,
then running the provided installer::

    $ git clone git@github.com:QInfer/python-qinfer.git
    $ cd python-qinfer
    $ pip install -r requirements.txt
    $ python setup.py install

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

.. _Anaconda distribution: https://www.continuum.io/downloads
.. _Sphinx: http://sphinx-doc.org/
