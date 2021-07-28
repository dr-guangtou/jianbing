#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from . import hsc
from . import venn
from . import split
from . import utils
from . import cutout
from . import visual
from . import profile
from . import prepare
from . import scatter
from . import catalog
from . import wlensing
from . import simulation

__all__ = ["hsc", "venn", "split", "utils", "cutout", "wlensing", "catalog",
           "catalog", "profile", "prepare", "visual", "simulation"]

HOME = os.environ['HOME']
DATA_DIR = os.path.join(HOME, "astro1/hsc/jianbing")

JB_DIR = os.path.join(HOME, "Dropbox/work/submit/jianbing")

SIM_DIR = os.path.join(JB_DIR, "data", "simulation")
RES_DIR = os.path.join(JB_DIR, "data", "results")
BIN_DIR = os.path.join(JB_DIR, "data", "bins")
FIG_DIR = os.path.join(JB_DIR, "paper", "figure")
