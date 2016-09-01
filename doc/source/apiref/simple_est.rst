..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _simple_est:
    
.. currentmodule:: qinfer

Simple Estimation
=================

Function Reference
------------------

.. autofunction:: simple_est_prec

.. autofunction:: simple_est_rb

.. _simple_est_data_arg:

Data Argument Type
------------------

Each of the functions above takes as its first argument
the data to be used in estimation. This data can be passed
in two different ways (more will be added soon):

- A `file-like object`_ or a `str` containing a file name:
  These will cause the data to be loaded from the given file
  as comma-separated values. Columns will be read in based on the
  order in which they appear in the file.
- A :class:`~pandas.DataFrame`: This will cause the data to be loaded
  from the given data frame, reading in columns by their headings.
- An :class:`~numpy.ndarray` with scalar data type and shape ``(n_rows, n_cols)``:
  Each column will be read in by its order.
- An :class:`~numpy.ndarray` with record data types and shape ``(n_rows, )``:
  Each column will be read in as a field of the array.

.. _file-like object: https://docs.python.org/3/glossary.html#term-file-object

.. _simple_est_extra_return:

Extra Return Values
-------------------

Each of the functions above supports an argument ``return_all``. If
`True`, a dictionary with the following fields will be returned as well:

- ``updater`` (:class:`~qinfer.SMCUpdater`): An updater representing
  the final posterior for the estimation procedure.
 