#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# __init__.py: Root of Qinfer package.
##
# © 2012 Chris Ferrie (csferrie@gmail.com) and
#        Christopher E. Granade (cgranade@gmail.com)
#     
# This file is a part of the Qinfer project.
# Licensed under the AGPL version 3.
##
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
##

from __future__ import absolute_import
from qinfer.version import version as __version__

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

import qinfer.tomography

