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
# to avoid triggering
# "Fatal IO error 11 (Resource temporarily unavailable) on X server"
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

def pretty_time(secs, force_h=False, force_m=False):
    if secs > 86400:
        return "{d} days, ".format(int(secs//86400)) + pretty_time(secs % 86400, force_h=True)
    elif force_h or secs > 3600:
        return "{h}:".format(h=int(secs//3600)) + pretty_time(secs % 3600, force_m=True)
    elif force_m or secs > 60:
        return (
            "{m:0>2}:{s:0>2}" if force_m else "{m}:{s:0>2}"
        ).format(m=int(secs//60), s=int(secs%60))
    else:
        return "{s} seconds".format(s=int(secs))

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
    "GET", "SET", "COMPLETE", "CLOSE"
)

## CLASSES #####################################################################

class ProgressDialog(object):
    # TODO: docstring
    
    def __init__(self,
            task_title="Progress",
            task_status="Waiting...",
            maxprog=100,
            on_cancel=None,
            eta=False
        ):
        
        # Make environment variables including the path to qinfer.
        env = dict(os.environ) # Copy so we don't make changes.
        try:
            import qinfer
            qinfer_path = os.path.split(qinfer.__file__)[0]
            if "PYTHONPATH" in env:
                env["PYTHONPATH"] += os.pathsep + qinfer_path
            else:
                env["PYTHONPATH"] = qinfer_path
        except ImportError:
            print "[WARN] Importing qinfer didn't work... progress tracking may fail."
        
        # Start the GUI in a new process.
        self._listener, self._port = _get_conn()
        self._process = subprocess.Popen(
            (sys.executable, "-m", "qinfer.dialogs", str(self._port), str(eta)),
            stdout=subprocess.PIPE, stdin=subprocess.PIPE, env=env
        )
        
        self._conn = self._listener.accept()
        
        self._on_cancel = on_cancel
        # TODO: implement cancel callback.
        
        self.title = task_title
        self.status = task_status
        self.maxprog = maxprog
        
    def __del__(self):
        self.close()
        
    def close(self):
        try:
            self._conn.send((Actions.CLOSE, None, None))
        except:
            pass
        time.sleep(0.02)
        try:
            self._process.poll()
        except:
            pass
        if self._process.returncode is None: # proc is still alive
            self._process.terminate()
            
    def complete(self):
        self._conn.send((Actions.COMPLETE, None, None))
    
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
    
    from .config import read_config, save_config
    
    # These imports are dangerous in a multiprocess environment,
    # so hide them here.
    from .ui import progbar as ui_progbar
    from PySide import QtGui, QtCore
    import numpy as np
    import time
    
    ## CLASSES #################################################################
    
    class _ProgressDialog(QtGui.QDialog):
        """
        This class is the "real" progress dialog, so that all GUI logic can be
        isolated to a different process.
        """
        
        def __init__(self, parent=None, eta=False):
            super(_ProgressDialog, self).__init__(parent)
            self.ui = ui_progbar.Ui_ProgBarDialog()
            self.ui.setupUi(self)
            
            self._eta = eta
            if eta:
                self._eta_history_len = 100 # How many samples back to recall.
                self._ts = np.zeros(self._eta_history_len)
                self._ps = np.zeros(self._eta_history_len)
                self._n_hist = 0
                self.ui.lbl_eta.show()
            else:
                self.ui.lbl_eta.hide()

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
            if self._eta:
                self._record_time()
                
        def _record_time(self):
            t = time.time()
            p = self.task_progress
            
            if self._n_hist < self._eta_history_len:
                self._ts[self._n_hist] = t
                self._ps[self._n_hist] = p
            else:
                # Cycle back the buffer.
                self._ts[:-1] = self._ts[1:]
                self._ps[:-1] = self._ps[1:]
                self._ts[-1] = t
                self._ps[-1] = p
                
            self._n_hist += 1
            
            # Re-estimate the ETA.
            n = min(self._eta_history_len, self._n_hist)
            if n >= 2:
                dp_dt = (self._ps[1:n] - self._ps[:n-1]) / (self._ts[1:n] - self._ts[:n-1])
                rate = np.mean(dp_dt)
                eta = (self.max_progress - self.task_progress) / rate
                self.ui.lbl_eta.setText(
                    "Estimated time remaining: {}".format(
                        pretty_time(eta)
                    )
                )
            else:
                self.ui.lbl_eta.setText(
                    "Estimated time remaining: computing..."
                )
                

    ## FUNCTIONS ###############################################################

    def do_completion():
        """
        Called when a task completes. Responsible for calling any
        command-line programs the user has specified to be triggered on task
        completions.
        """
        config = read_config()
        if config.has_option("Task Monitoring", "completion_cmd"):
            call_cmd = config.get("Task Monitoring", "completion_cmd")
            subprocess.call(
                call_cmd.format(
                    task_title=dialog.task_title
                ),
                shell=True) # FIXME: shouldn't use shell.

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
        elif action == Actions.COMPLETE:
            do_completion()
            
    ## SCRIPT ##################################################################     

    # Make a new conn, since we're emulating multiprocessing
    # in a forkless way.
    port = int(sys.argv[1])
    eta  = bool(sys.argv[2])
    conn = multiprocessing.connection.Client(('localhost', port),
        authkey='notreallysecret'
    )

    # NOW go on and make the GUI!
    # Window name set using hack at:
    # https://groups.google.com/forum/#!topic/pyside/24qxvwfrRDs
    app = QtGui.QApplication(["Progress"] + sys.argv[1:])
    app.setApplicationName("Progress")
    
    # TODO: Change which dialog based on other command line arguments.
    dialog = _ProgressDialog(eta=eta)
    dialog.show()
    
    idle_timer = QtCore.QTimer()
    idle_timer.setInterval(0)
    idle_timer.timeout.connect(handle_conn)
    idle_timer.start()
    
    app.exec_()
