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

## NOTES #######################################################################

# Several parts of the design of this modules are based on
# http://stackoverflow.com/questions/7714868/python-multiprocessing-how-can-i-reliably-redirect-stdout-from-a-child-process/11779039#11779039
# to avoid triggering "Fatal IO error 11 (Resource temporarily unavailable) on X server"
# due to multiple processes sharing X resources.

## FEATURES ####################################################################

from __future__ import division

## ALL #########################################################################

# We use __all__ to restrict what globals are visible to external modules.
__all__ = [
    'ProgressDialog'
]

## IMPORTS #####################################################################

import sys, os
import time
import multiprocessing.connection
import subprocess
import socket
import warnings
from itertools import count

from ._lib import enum

## FUNCTIONS ###################################################################

def _get_conn():
    for port in count(10000):
        try:
            listener = multiprocessing.connection.Listener(
                ('localhost', int(port)),
                authkey='notreallysecret'
            )
            return listener, port
        except socket.error as ex:
            if ex.errno != 98: # Err 98 is port not available.
                raise ex
    
   
## ENUMS #######################################################################

Properties = enum.enum(
    "TITLE", "STATUS", "MAXPROG", "PROGRESS",
    # TODO: max/min progress
)
    
Actions = enum.enum(
    "GET", "SET", "CLOSE"
)

## CLASSES #####################################################################

class ProgressDialog(object):
    # TODO: docstring
    
    def __init__(self,
            task_title="Progress",
            task_status="Waiting...",
            maxprog=100,
            on_cancel=None
        ):
        
        # Start the GUI in a new process.
        self._listener, self._port = _get_conn()
        self._process = subprocess.Popen(
            (sys.executable, "-m", "qinfer.dialogs", str(self._port)),
            stdout=subprocess.PIPE, stdin=subprocess.PIPE
        )
        
        self._conn = self._listener.accept()
        
        self._on_cancel = on_cancel
        # TODO: implement cancel callback.
        
        self.title = task_title
        self.status = task_status
        self.maxprog = maxprog
        
    def __del__(self):
        self._conn.send((Actions.CLOSE, None, None))
        time.sleep(0.02)
        self._process.poll()
        if self._process.returncode is None: # proc is still alive
            self._process.terminate()
    
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
    
    
## MAIN CODE ###################################################################
# Everything here only gets run when this module is run as a script, thus
# ensuring that it doesn't get called from within other programs.
# This way, the X11 connection created never existed in any other process.
if __name__ == "__main__":
    
    ## IMPORTS #################################################################
    
    # These imports are dangerous in a multiprocess environment,
    # so hide them here.
    from .ui import progbar as ui_progbar
    from PySide import QtGui, QtCore
    
    ## CLASSES #################################################################
    
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

    ## FUNCTIONS ###############################################################

    def handle_conn():
        # Recieve one thing from the conn, then return.
        # Since this is handled in the idle loop, we'll
        # get to pull another thing later.
        #
        # Note that conn is a global.
        if not conn.poll(0.01): # Wait for 10ms.
            # Nothing to read. Try again later.
            return
            
        action, prop, newval = conn.recv()
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
        elif action == Actions.CLOSE:
            app.exit()
            
    ## SCRIPT ##################################################################     

    # Make a new conn, since we're emulating multiprocessing
    # in a forkless way.
    port = int(sys.argv[1])
    conn = multiprocessing.connection.Client(('localhost', port),
        authkey='notreallysecret'
    )

    # NOW go on and make the GUI!
    app = QtGui.QApplication(sys.argv)
    
    dialog = _ProgressDialog()
    dialog.show()
    
    idle_timer = QtCore.QTimer()
    idle_timer.setInterval(0)
    idle_timer.timeout.connect(handle_conn)
    idle_timer.start()
    
    app.exec_()
