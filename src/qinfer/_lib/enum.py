#!/usr/bin/python
# -*- coding: utf-8 -*-
##
# enum.py: Pythonic enumerations.
##
# (c) 2013 StackExchange.
# Licensed under Creative Commons BY-SA 3.0:
#     http://creativecommons.org/licenses/by-sa/3.0/
# This module is based on the answer at
#     http://stackoverflow.com/a/1695250
# by Alec Thomas.
##
from __future__ import absolute_import

from builtins import range
from future.utils import iteritems


def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    reverse = dict((value, key) for key, value in iteritems(enums))
    enums['reverse_mapping'] = reverse
    return type('Enum', (), enums)
