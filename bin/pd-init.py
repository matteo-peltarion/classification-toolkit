#!/usr/bin/env python

import os

import palladio

import shutil

template_file = os.path.join(
    os.path.dirname(palladio.__file__),
    "config", "konfiguration.template.py")

shutil.copy(template_file, "konfiguration.py")
