..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _intro:
    
Introduction
============

TODO

Obtaining QInfer
----------------

A stable version of **QInfer** has not yet been released. Until then,
the latest version may always be obtained by cloning into the GitHub
repository::

    $ git clone git@github.com:csferrie/python-qinfer.git
    
Once obtained in this way, **QInfer** may be updated by pulling from GitHub::

    $ git pull

Installation
------------

**QInfer** provides a setup script, ``setup.py``, for installing from source.
On Unix-like systems, **QInfer** can be made available globally by running::

    $ cd /path/to/qinfer/
    $ sudo python setup.py install

On Windows, run ``cmd.exe``, then run the setup script::

    C:\> cd C:\path\to\qinfer\
    C:\path\to\qinfer\> python setup.py install
    
Note that you may be prompted for permission by User Access Control.

Dependencies
------------

**QInfer** depends on only a very few packages:

- Python 2.7 (may work with earlier, but not tested).
- NumPy and SciPy.

Some features of **QInfer** require additional packages, but the core of
**QInfer** will work without them:

- `SciKit-Learn`_ required for some advanced features.
- `Sphinx`_ required to rebuild documentation.
- `matplotlib`_ required for plotting functionality.
- `IPython Notebook`_ (version 1.1) is used to provide examples.

On Windows, these packages can be provided by `Python(x,y)`_. Linux users may
obtain the needed dependencies using package managers. Under Ubuntu::

    $ sudo apt-get install python2.7 python-numpy python-scipy python-scikits-learn python-matplotlib python-sphinx ipython-notebook
    
On Fedora::

    $ sudo yum install python numpy scipy python-sphinx python-matplotlib python-ipython-notebook
    $ sudo easy_install -U scikit-learn

Alternatively,
`Enthought Canopy`_ has been tested with **QInfer**, and may be
used on Windows, Mac OS X or Linux.

Citing QInfer
-------------

If you use **QInfer** in your publication or presentation, we would appreciate it
if you cited our work. We recommend citing **QInfer** by using the BibTeX
entry::

    @Misc{,
      author = {Christopher Ferrie and Christopher Granade and others},
      title =  {{QInfer}: Library for Statistical Inference in Quantum Information},
      year =   {2012--},
      url =    "https://github.com/csferrie/python-qinfer"
    }

To indicate which version of **QInfer** you used in your work, it may be helpful
to cite a given SHA hash as listed on
`GitHub <https://github.com/csferrie/python-qinfer/commits/master>`_ (the
hashes of each commit are listed on the right hand side of the page).
A recommended BibTeX entry for citing a particular version is::

    @Misc{,
      author = {Christopher Ferrie and Christopher Granade and others},
      title =  {{QInfer}: Library for Statistical Inference in Quantum Information},
      year =   {2013},
      month =  {2},
      day =    {18},
      url =    "https://github.com/csferrie/python-qinfer/commit/d04bc1d53933f13065917c15ccb0e2f127de3b8a",
      note =   {Version \texttt{d04bc1d53933f13065917c15ccb0e2f127de3b8a}.}
    }
    
In this example, ``d04bc1d53933f13065917c15ccb0e2f127de3b8a`` should be replaced by the
particular version being cited.

Getting Started
---------------

TODO

.. _Enthought Canopy: https://www.enthought.com/products/canopy/
.. _Python(x,y): http://code.google.com/p/pythonxy/
.. _matplotlib: http://matplotlib.org/
.. _SciKit-Learn: http://scikit-learn.org/stable/
.. _Sphinx: http://sphinx-doc.org/
.. _IPython Notebook: http://ipython.org/ipython-doc/stable/interactive/notebook.html
