=================
Welcome to QInfer
=================

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.51273.svg
   :target: http://dx.doi.org/10.5281/zenodo.51273

.. image:: https://img.shields.io/badge/launch-binder-E66581.svg
    :target: http://mybinder.org/repo/qinfer/qinfer-examples
    :alt: Launch Binder
    
.. image:: https://img.shields.io/pypi/v/QInfer.svg?maxAge=2592000
    :target: https://pypi.python.org/pypi/QInfer
    

.. image:: https://travis-ci.org/QInfer/python-qinfer.svg?branch=master
    :target: https://travis-ci.org/QInfer/python-qinfer

.. image:: https://coveralls.io/repos/github/QInfer/python-qinfer/badge.svg?branch=master
    :target: https://coveralls.io/github/QInfer/python-qinfer?branch=master 

.. image:: https://codeclimate.com/github/QInfer/python-qinfer/badges/gpa.svg
   :target: https://codeclimate.com/github/QInfer/python-qinfer
   :alt: Code Climate

**QInfer** is a library using Bayesian sequential Monte Carlo for quantum
parameter estimation. Works with Python 2.7, 3.3, 3.4 and 3.5.

Installing QInfer
=================

We recommend using **QInfer** with the
`Anaconda distribution`_. Download and install
Anaconda for your platform, either Python 2.7 or 3.5. We
suggest using Python 3.5, but **QInfer**
works with either. Next, ensure that you have Git installed. On Windows,
we suggest the `official Git downloads <https://git-scm.com/downloads>`_.
Once Anaconda and Git are installed, simply run ``pip`` to install **QInfer**::

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
