..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _intro:
    
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
works with either. Once Anaconda is installed, simply run ``pip`` to install **QInfer**::

    $ pip install git+https://github.com/QInfer/python-qinfer.git

Alternatively, **QInfer** can be installed manually by downloading from GitHub,
then running the provided installer::

    $ git clone git@github.com:QInfer/python-qinfer.git
    $ cd python-qinfer
    $ pip install -r requirements.txt
    $ python setup.py install

Citing QInfer
-------------

If you use **QInfer** in your publication or presentation, we would appreciate it
if you cited our work. We recommend citing **QInfer** by using the BibTeX
entry::

    @Misc{,
      author = {Christopher Granade and Christopher Ferrie and others},
      title =  {{QInfer}: Library for Statistical Inference in Quantum Information},
      year =   {2012--},
      url =    "https://github.com/QInfer/python-qinfer"
    }

To indicate which version of **QInfer** you used in your work, it may be helpful
to cite a given SHA hash as listed on
`GitHub <https://github.com/QInfer/python-qinfer/commits/master>`_ (the
hashes of each commit are listed on the right hand side of the page).
A recommended BibTeX entry for citing a particular version is::

    @Misc{,
      author = {Christopher Granade and Christopher Ferrie and others},
      title =  {{QInfer}: Library for Statistical Inference in Quantum Information},
      year =   {2012},
      month =  {2},
      day =    {18},
      url =    "https://github.com/QInfer/python-qinfer/commit/d04bc1d53933f13065917c15ccb0e2f127de3b8a",
      note =   {Version \texttt{d04bc1d53933f13065917c15ccb0e2f127de3b8a}.}
    }
    
In this example, ``d04bc1d53933f13065917c15ccb0e2f127de3b8a`` should be replaced by the
particular version being cited.

Getting Started
---------------

To get started using **QInfer**, it may be helpful to give a look through the
:ref:`guide`. Alternatively, you may want to dive right into looking at
the `examples`_ using `IPython Notebook`_. To do so, from your system command
line, navigate to where you downloaded **QInfer**, then run::

    cd python-qinfer/examples
    ipython notebook

Alternatively, IPython provides an online viewer for reading notebook files:

- `Plotting Example <http://nbviewer.ipython.org/github/csferrie/python-qinfer/blob/master/examples/plot_example.ipynb>`_
- `Noisy Coin Example <http://nbviewer.ipython.org/github/csferrie/python-qinfer/blob/master/examples/Noisy%20Coin%20Example.ipynb>`_

More details can be found in the :ref:`examples` section.

.. _Anaconda distribution: https://www.continuum.io/downloads
.. _Sphinx: http://sphinx-doc.org/
.. _IPython Notebook: http://ipython.org/ipython-doc/stable/interactive/notebook.html

.. _examples: https://github.com/csferrie/python-qinfer/tree/master/examples
