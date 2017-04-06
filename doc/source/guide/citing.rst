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

Alternatively, it may be more reliable to use environment variables to turn
on `duecredit`_ collection, since this approach also works with 
Jupyter Notebook::

    $ export DUECREDIT_ENABLE=yes # Bash
    PS> $Env:DUECREDIT_ENABLE = "yes" # PowerShell
    
Or, from within an Jupyter Notebook, environment variables can be set before
import using::

    import os
    os.environ['DUECREDIT_ENABLE'] = 'yes'
    import qinfer as qi

Whenever it is detected that this variable is true, any uses of **Qinfer** in a 
given folder (even multiple distinct runs) will maintain 
a file in the same folder called ``.duecredit.p`` that contains 
a representation of your bibliography. This file is updated whenever **Qinfer**
uses a module, class, or method that is appropriately tagged with a citation.

If you wish to see the citation compilaton of a *single* python session, 
you can dump the current state with::

    qi.due.dump()

On the other hand, to print it out the entire ``.duecredit.p`` collection 
in BibTeX form, use the summary functionality of `duecredit`_::

    $ duecredit summary --format=bibtex

Note that this summary will also include projects such as NumPy and SciKit-Learn
that are supported by `duecredit`_, as well as any other projects which natively
host citation metadata through `duecredit`_.  For example::

    $ export DUECREDIT_ENABLE=yes
    $ ipython
    In [1]: import qinfer as qi
    In [2]: exit

    DueCredit Report:
    - Scientific tools library / numpy (v 1.11.1) [1]
    - Bayesian inference for quantum information / qinfer (v 1.0) [2]
    - Machine Learning library / sklearn (v 0.17.1) [3]
        - Affinity propagation clustering algorithm / sklearn.cluster.affinity_propagation_ (v 0.17.1) [4]

    3 packages cited
    1 module cited
    0 functions cited

    References
    ----------

    [1] Van Der Walt, S., Colbert, S.C. & Varoquaux, G., 2011. The NumPy array: a structure for efficient numerical computation. Computing in Science & Engineering, 13(2), pp.22–30.
    [2] Granade, C. et al., 2016. QInfer: Statistical Inference Software for Quantum Applications. arXiv:1610.00336 [physics, physics:quant-ph, stat].
    [3] Pedregosa, F. et al., 2011. Scikit-learn: Machine learning in Python. The Journal of Machine Learning Research, 12, pp.2825–2830.
    [4] Frey, B.J. & Dueck, D., 2007. Clustering by Passing Messages Between Data Points. Science, 315(5814), pp.972–976.

The bibliography entries defined by QInfer are organized according to different
*tags*, making it easier to filter through the results of `duecredit`_. In particular,
QInfer uses the following citation tags, as defined in the `duecredit documentation
<https://github.com/duecredit/duecredit/blob/master/README.md>`_:

- ``implementation``: The tagged function is an implementation of the cited work.
- ``experiment``: Concerns experimental demonstrations of an algorithm or procedure. This tag
  is similar to, but distinct from, the ``use`` tag defined by `duecredit`_.

These tags can be controlled using the ``DUECREDIT_REPORT_ALL`` and ``DUECREDIT_REPORT_TAGS``
environment variables. By default, all tags by ``implementation`` are hidden, such that
summaries of the collected bibliography describe which software implementations are used
in the course of a project.

For more details on how to use `duecredit`_, please see their `documentation on
GitHub <https://github.com/duecredit/duecredit/blob/master/README.md>`_.


.. _duecredit: https://github.com/duecredit/duecredit/
