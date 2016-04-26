#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# config.py: Stores configuration information.
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
    'read_config',
    'save_config',
]

## TODO #######################################################################

# Put section header names here.

## IMPORTS ####################################################################

import sys, os
import configparser

## FUNCTIONS ##################################################################

def ensuredir(filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def preffilename():
    if sys.platform.startswith('linux') or sys.platform.startswith('darwin') or sys.platform.startswith('cygwin'):	    
        # Unix-y
        configfile = os.path.expanduser('~/.config/qinfer.conf')
    elif sys.platform.startswith('win'):
        # Windows-y
        configfile = os.path.join(os.path.expandvars('%APPDATA%'), 'qinfer.conf')
    else:
        return NotImplemented

    ensuredir(configfile)
    return configfile
    
def read_config():
    #
    parser = configparser.SafeConfigParser()
    parser.read(preffilename())
    return parser
    
def save_config(parser):
    with open(preffilename(), 'w') as f:
        parser.write(f)

    
