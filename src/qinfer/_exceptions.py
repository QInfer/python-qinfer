#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# _exceptions.py: Derived classes for Errors and Warnings specific to Qinfer.
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

## FEATURES ###################################################################

from __future__ import absolute_import
from __future__ import division

## ALL ########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
    'ResamplerWarning', 'ResamplerError', 'ApproximationWarning'
]

## IMPORTS ####################################################################

import warnings

## CLASSES ####################################################################

class ResamplerError(RuntimeError):
    """
    Error failed when a resampler has failed in an unrecoverable manner.
    """
    def __init__(self, msg, cause=None):
        super(ResamplerError, self).__init__(
            "{}, caused by exception: {}".format(msg, cause)
            if cause is not None else msg
        )
        self._cause = cause

class ResamplerWarning(RuntimeWarning):
    """
    Warning raised in response to events within resampling steps.
    """
    pass
    
class ApproximationWarning(RuntimeWarning):
    """
    Warning raised when a numerical approximation fails in a way that may
    violate assumptions, such as when a negative variance is observed due to
    numerical errors.
    """
    pass

