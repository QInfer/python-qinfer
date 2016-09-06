..
    This work is licensed under the Creative Commons Attribution-
    NonCommercial-ShareAlike 3.0 Unported License. To view a copy of this
    license, visit http://creativecommons.org/licenses/by-nc-sa/3.0/ or send a
    letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View,
    California, 94041, USA.
    
.. _interop_guide:
    
.. currentmodule:: qinfer

Interoperability
================

Introduction
------------

**QInfer** can be used in conjunction with software written in scientific
software platforms other than Python, thanks to the ready availability of
interoperability libraries for interacting with Python. In this section,
we provide brief examples of using these libraries with **QInfer**.

MATLAB Interoperability
-----------------------

As of version 2016a, MATLAB includes `built-in functions for calling
Python-language software <www.mathworks.com/help/matlab/call-python-libraries.html>`_.
In particular, these functions can be used to use **QInfer** from within MATLAB.
For example, the following MATLAB snippet will generate and analyze frequency estimation
data using the :ref:`simple_est` functions provided by **QInfer**.

.. code:: matlab

    >> true_omega = 70.3;
    >> n_shots = 400;
    >> 
    >> ts = pi * (1:1:100) / (2 * 100);
    >> 
    >> signal = sin(true_omega * ts / 2) .^ 2;
    >> counts = binornd(n_shots, signal);
    >> 
    >> setenv MKL_NUM_THREADS 1
    >> data = py.numpy.column_stack({counts ts ...
    n_shots * ones(1, size(ts, 2))});
    >> est = py.qinfer.simple_est_prec(data, ...
    pyargs('freq_min', 0, 'freq_max', 100));

Importantly, the ``setenv`` command is *required* to work around
a bug internal to the MATLAB interpreter.

Julia Interoperability
----------------------

In Julia, interoperability can be achieved using the `PyCall.jl`_
package, which provides macros for making Python modules
available as Julia variables. To install PyCall.jl, use
Julia's built-in package installer:

.. code:: julia

    julia> Pkg.add("PyCall")

After installing PyCall.jl, the example above proceeds
in a very similar fashion:

.. code:: julia

    julia> true_omega = 70.3
    julia> n_shots = 100
    julia> 
    julia> ts = pi * (1:1:100) / (2 * 100)
    julia> 
    julia> signal = sin(true_omega * ts / 2) .^ 2
    julia> counts = map(p -> rand(Binomial(n_shots, p)), signal);
    julia> @pyimport numpy as np
    julia> @pyimport qinfer as qi
    julia> 
    julia> data = [counts'; ts'; n_shots * ones(length(ts))']'
    julia> est_mean, est_cov = qi.simple_est_prec(data, freq_min=0, freq_max=100)

.. _PyCall.jl: https://github.com/stevengj/PyCall.jl