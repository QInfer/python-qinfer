#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# _exceptions.py: Derived classes for Errors and Warnings specific to QInfer.
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

