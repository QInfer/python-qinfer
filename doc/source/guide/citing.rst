..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _citing_guide:
    
.. currentmodule:: qinfer

Citing QInfer and Related Projects
==================================

Citing Released Versions
------------------------

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


Pre-Release Versions
--------------------

If you wish to cite **QInfer** functionality that has not yet appeared in a
released version, it may be helpful
to cite a given SHA hash as listed on
`GitHub <https://github.com/QInfer/python-qinfer/commits/master>`_ (the
hashes of each commit are listed on the right hand side of the page).
A recommended BibTeX entry for citing a particular commit is::

    @misc{qinfer-prerelease,
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

Automatic Citation Collection
-----------------------------

QInfer also supports the use of the `duecredit`_ project to automatically
collect citations for a bibliography. This support is still experimental,
and may not yet generate complete bibliographies, so please manually check
the resulting bibliography. In any case, to get started, install `duecredit`_::

    $ pip install duecredit

If your project then uses a script to generate and/or analyze data, then you
can use `duecredit`_ to collect citations for Python modules and functions
called by your script. For example, if your project can be run as
`python script.py`, use the following command to collect a bibliography::

    $ python -m duecredit script.py

This will create a file called ``.duecredit.p`` containing a representation
of your bibliography. To print it out in BibTeX form, use the summary functionality
of `duecredit`_::

    $ duecredit summary --format=bibtex

For more details on how to use `duecredit`_, please see their `documentation on
GitHub <https://github.com/duecredit/duecredit/blob/master/README.md>`_.


.. _duecredit: https://github.com/duecredit/duecredit/
