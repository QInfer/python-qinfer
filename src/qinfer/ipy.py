#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# ipy.py: Interaction with IPython and Jupyter.
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

## EXPORTS ###################################################################

__all__ = ['IPythonProgressBar']

## IMPORTS ####################################################################

try:
    from IPython.display import display
    import ipywidgets as ipw
except:
    display = None
    ipw = None

## CLASSES ###################################################################

class IPythonProgressBar(object):
    """
    Represents a progress bar as an IPython widget. If the widget
    is closed by the user, or by calling ``finalize()``, any further
    operations will be ignored.

    .. note::

        This progress bar is compatible with QuTiP progress bar
        classes.
    """
    def __init__(self):
        if ipw is None:
            raise ImportError("IPython support requires the ipywidgets package.")
        self.widget = ipw.FloatProgress(
            value=0.0, min=0.0, max=100.0, step=0.5,
            description=""
        )

    @property
    def description(self):
        """
        Text description for the progress bar widget,
        or ``None`` if the widget has been closed.

        :type: `str`
        """
        try:
            return self.widget.description
        except:
            return None
    @description.setter
    def description(self, value):
        try:
            self.widget.description = value
        except:
            pass

    def start(self, max):
        """
        Displays the progress bar for a given maximum value.

        :param float max: Maximum value of the progress bar.
        """
        try:
            self.widget.max = max
            display(self.widget)
        except:
            pass

    def update(self, n):
        """
        Updates the progress bar to display a new value.
        """
        try:
            self.widget.value = n
        except:
            pass

    def finished(self):
        """
        Destroys the progress bar.
        """
        try:
            self.widget.close()
        except:
            pass
