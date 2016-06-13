#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# ipy.py: Interaction with IPython and Jupyter.
##
# © 2016 Chris Ferrie (csferrie@gmail.com) and
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
