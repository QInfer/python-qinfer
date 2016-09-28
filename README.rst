=================
Welcome to QInfer
=================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.157007.svg
   :target: https://doi.org/10.5281/zenodo.157007

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
works with either.

If using Anaconda, you should go ahead now and install from their repository
all the dependencies that you can. If you are using "regular" Python then you can
ignore this step. Replace ``python=3.5`` with your version (typically
either 2.7 or 3.5).

.. code-block:: console

    $ conda install python=3.5 numpy scipy matplotlib scikit-learn

If you are **not** using Anaconda, but are instead using "regular" Python,
and you are on Linux, you will need the Python development package:

.. code-block:: console

    $ sudo apt-get install python-dev

Where ``python-dev`` might be ``python3.5-dev`` depending on your package
manager and which version of Python you are using.

The latest release of **QInfer** can now be installed from
``PyPI`` with ``pip``:

.. code-block:: console

    $ pip install qinfer

Alternatively, **QInfer** can be installed using ``pip`` and Git. Ensure that
you have Git installed. On Windows, we suggest the
`official Git downloads <https://git-scm.com/downloads>`__.
Once Anaconda and Git are installed, simply run ``pip`` to install **QInfer**:

.. code-block:: console

    $ pip install git+https://github.com/QInfer/python-qinfer.git

Lastly, **QInfer** can be installed manually by downloading from GitHub,
then running the provided installer:

.. code-block:: console

    $ git clone git@github.com:QInfer/python-qinfer.git
    $ cd python-qinfer
    $ pip install -r requirements.txt
    $ python setup.py install

More Information
================

Full documentation for **QInfer** is
`available on ReadTheDocs <http://docs.qinfer.org/>`_,
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
