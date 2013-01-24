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

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    reverse = dict((value, key) for key, value in enums.iteritems())
    enums['reverse_mapping'] = reverse
    return type('Enum', (), enums)

