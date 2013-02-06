#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# dialogs.py: Common dialog boxes.
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

## FEATURES ####################################################################

from __future__ import division

## ALL #########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
    'ProgressDialog'
]

## IMPORTS #####################################################################

import sys
import multiprocessing
import warnings

from .ui import progbar as ui_progbar
from ._lib import enum

from PySide import QtGui, QtCore

   
## ENUMS #######################################################################

Properties = enum.enum(
    "TITLE", "STATUS", "MAXPROG", "PROGRESS",
    # TODO: max/min progress
)
    
Actions = enum.enum(
    "GET", "SET"
)

## CLASSES #####################################################################

class _ProgressDialog(QtGui.QDialog):
    """
    This class is the "real" progress dialog, so that all GUI logic can be
    isolated to a different process.
    """
    
    def __init__(self, parent=None):
        super(_ProgressDialog, self).__init__(parent)
        self.ui = ui_progbar.Ui_ProgBarDialog()
        self.ui.setupUi(self)

    @property
    def task_title(self):
        return str(self.ui.lbl_task_title.text())
    @task_title.setter
    def task_title(self, newval):
        newval = str(newval)
        self.setWindowTitle(newval)
        self.ui.lbl_task_title.setText(newval)
        
    @property
    def task_status(self):
        return str(self.ui.lbl_task_status.text())
    @task_status.setter
    def task_status(self, newval):
        newval = str(newval)
        self.ui.lbl_task_status.setText(newval)
     
    @property
    def max_progress(self):
        return int(self.ui.prog_bar.maximum())
    @max_progress.setter
    def max_progress(self, newval):
        self.ui.prog_bar.setMaximum(int(newval))
        
    @property
    def task_progress(self):
        return int(self.ui.prog_bar.value())
    @task_progress.setter
    def task_progress(self, newval):
        self.ui.prog_bar.setValue(int(newval))

def _process_start(child_conn):

    def handle_conn():
        # Recieve one thing from the conn, then return.
        # Since this is handled in the idle loop, we'll
        # get to pull another thing later.
        if not child_conn.poll(0.01): # Wait for 10ms.
            # Nothing to read. Try again later.
            return
            
        action, prop, newval = child_conn.recv()
        if action == Actions.SET:
            if prop == Properties.TITLE:
                dialog.task_title = newval
            elif prop == Properties.STATUS:
                dialog.task_status = newval
            elif prop == Properties.MAXPROG:
                dialog.max_progress = newval
            elif prop == Properties.PROGRESS:
                dialog.task_progress = newval
        elif action == Actions.GET:
            # FIXME: ignored for now.
            pass
        

    app = QtGui.QApplication(sys.argv)
    
    dialog = _ProgressDialog()
    dialog.show()
    
    idle_timer = QtCore.QTimer()
    idle_timer.setInterval(0)
    idle_timer.timeout.connect(handle_conn)
    idle_timer.start()
    
    app.exec_()

class ProgressDialog(object):
    # TODO: docstring
    
    def __init__(self,
            task_title="Progress",
            task_status="Waiting...",
            maxprog=100,
            on_cancel=None
        ):
        
        # Create a parent and child connection pipe.
        parent_conn, child_conn = multiprocessing.Pipe()
        self._conn = parent_conn
        
        # Start the GUI in a new process.
        self._process = multiprocessing.Process(
            target=_process_start, args=(child_conn,)
        )
        self._process.start()
        
        self._on_cancel = on_cancel
        # TODO: implement cancel callback.
        
        self.title = task_title
        self.status = task_status
        self.maxprog = maxprog
        
    def __del__(self):
        if self._process.is_alive():
            self._process.terminate() # FIXME: send a command to exit instead.
    
    # TODO: implement reading out progress, title and status.    
    title = property()
    @title.setter
    def title(self, newval):
        self._conn.send((Actions.SET, Properties.TITLE, newval))
    
    status = property()
    @status.setter
    def status(self, newval):
        self._conn.send((Actions.SET, Properties.STATUS, newval))
        
    maxprog = property()
    @maxprog.setter
    def maxprog(self, newval):
        self._conn.send((Actions.SET, Properties.MAXPROG, newval))
        
    progress = property()
    @progress.setter
    def progress(self, newval):
        self._conn.send((Actions.SET, Properties.PROGRESS, newval))
    
