..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _intro:

============
Introduction
============

**QInfer** is a library for working with sequential Monte Carlo methods for
parameter estimation in quantum information. **QInfer** will use your custom
experimental models to estimate properties of those models based on experimental
data.

Additionally, **QInfer** is designed for use with cutting-edge tools, such as
Python and IPython, making it easier to integrate with the rich community of
Python-based scientific software libraries.


Installing QInfer
=================

We recommend using **QInfer** with the
`Anaconda distribution`_. Download and install
Anaconda for your platform, either Python 2.7 or 3.5. We
suggest using Python 3.5, but **QInfer**
works with either.
Once Anaconda is installed, simply run ``pip`` to install **QInfer**::

    $ pip install qinfer

Alternatively, **QInfer** can be installed manually by downloading from GitHub,
then running the provided installer::

    $ git clone git@github.com:QInfer/python-qinfer.git
    $ cd python-qinfer
    $ pip install -r requirements.txt
    $ python setup.py install

Citing QInfer
=============

If you use **QInfer** in your publication or presentation, we would appreciate it
if you cited our work. We recommend citing **QInfer** by using the BibTeX
entry::

    @misc{qinfer-1_0,
      author       = {Christopher Granade and
                      Christopher Ferrie and
                      Steven Casagrande and
                      Ian Hincks and
                      Michal Kononenko and
                      Thomas Alexander and
                      Yuval Sanders},
      title        = {{QInfer}: Library for Statistical Inference in Quantum Information},
      month        = september,
      year         = 2016,
      doi          = {10.5281/zenodo.157007},
      url          = {http://dx.doi.org/10.5281/zenodo.157007}
    }

If you wish to cite **QInfer** functionality that has not yet appeared in a
released version, it may be helpful
to cite a given SHA hash as listed on
`GitHub <https://github.com/QInfer/python-qinfer/commits/master>`_ (the
hashes of each commit are listed on the right hand side of the page).
A recommended BibTeX entry for citing a particular commit is::

    @misc{qinfer-1_0b4,
      author       = {Christopher Granade and
                      Christopher Ferrie and
                      Steven Casagrande and
                      Ian Hincks and
                      Michal Kononenko and
                      Thomas Alexander and
                      Yuval Sanders},
      title        = {{QInfer}: Library for Statistical Inference in Quantum Information},
      month        = may,
      year         = 2016,
      url =    "https://github.com/QInfer/python-qinfer/commit/bc3736c",
      note =   {Version \texttt{bc3736c}.}
    }

    
In this example, ``bc3736c`` should be replaced by the
particular commit being cited, and the date should be replaced by the date
of that commit.

Getting Started
===============

To get started using **QInfer**, it may be helpful to give a look through the
:ref:`guide`. Alternatively, you may want to dive right into looking at
some examples. We provide a number of `Jupyter Notebook`_-based examples
in the `qinfer-examples`_ repository. These examples can be viewed online
using `nbviewer`_, or can be run online using `binder`_ without installing any additional
software.

The examples can also be run locally, using the instructions available
at `qinfer-examples`_.

.. _Anaconda distribution: https://www.continuum.io/downloads
.. _Sphinx: http://sphinx-doc.org/
.. _Jupyter Notebook: http://jupyter.org/
.. _nbviewer: http://nbviewer.jupyter.org/github/qinfer/qinfer-examples/tree/master/
.. _binder: http://mybinder.org/repo/qinfer/qinfer-examples
.. _qinfer-examples: https://github.com/QInfer/qinfer-examples
