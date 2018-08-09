#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# __init__.py: Root of QInfer package.
##
# © 2017, Chris Ferrie (csferrie@gmail.com) and
#         Christopher Granade (cgranade@cgranade.com).
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the copyright holder nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
##

from __future__ import absolute_import
from qinfer.version import version as __version__

## CITATION METADATA ##########################################################

from ._due import due, Doi, BibTeX

due.cite(
    BibTeX("""
        @article{qinfer,
            title = {{QInfer}: {Statistical} Inference Software for Quantum Applications},
            eprinttype = {arxiv},
            eprint = {1610.00336},
            journal = {arXiv:1610.00336 [physics, physics:quant-ph, stat]},
            author = {Granade, Christopher and Ferrie, Christopher and Hincks, Ian and Casagrande, Steven and Alexander, Thomas and Gross, Jonathan and Kononenko, Michal and Sanders, Yuval},
            month = oct,
            year = {2016}
        }
    """),
    description="Bayesian inference for quantum information",
    tags=["implementation"],
    cite_module=True,
    path="qinfer"
)

## IMPORTS ####################################################################
# These imports control what is made available by importing qinfer itself.

from qinfer._exceptions import *

from qinfer.gpu_models import *
from qinfer.perf_testing import *
from qinfer.expdesign import *
from qinfer.test_models import *
from qinfer.distributions import *
from qinfer.abstract_model import *
from qinfer.parallel import *
from qinfer.score import *
from qinfer.rb import *
from qinfer.smc import *
from qinfer.unstructured_models import *
from qinfer.derived_models import *
from qinfer.ipy import *
from qinfer.simple_est import *
from qinfer.resamplers import *
from qinfer.domains import *

import qinfer.tomography

