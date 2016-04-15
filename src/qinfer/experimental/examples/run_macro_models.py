#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# run_macro_models.py: Runs examples of macro-based models.
##
# © 2014 Chris Ferrie (csferrie@gmail.com) and
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

## MACROPY ACTIVATION ########################################################

# The following line is needed to activate all MacroPy features. Nothing will
# work without this import. It's basically magic.
from __future__ import absolute_import
import macropy.activate

# We can export code as macros are applied, such that we don't need to depend
# on MacroPy to ship models made using these macros.
#
# This may generate errors if you cannot write to folders where macros are
# invoked; in particular, this will be the case if you have installed MacroPy
# to a site-packages directory.
# Until we find a good way to enable exporting in a robust manner, this will
# be commented out.
import macropy.core.exporters
#macropy.exporter = macropy.core.exporters.SaveExporter("exported")

# Finally, we can now import the module containing the macro-defined models.
# Since MacroPy has been activated, it will intercept this import using
# the New Import Hooks feature (PEP 302) and transform the module before
# executing it. The exporter will then save a copy of the transformed
# module.
import qinfer.experimental.examples.macro_models as mm

# We now can run the main() function defined there.
mm.main()
