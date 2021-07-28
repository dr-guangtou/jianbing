"""
Directory related to the TopN tests.
"""

# Please adjust these directories accordingly

import os

HOME = os.environ['HOME']

DATA_DIR = os.path.join(HOME, "astro1/hsc/jianbing")

JB_DIR = os.path.join(HOME, "Dropbox/work/submit/jianbing")

SIM_DIR = os.path.join(JB_DIR, "data", "simulation")
RES_DIR = os.path.join(JB_DIR, "data", "results")
BIN_DIR = os.path.join(JB_DIR, "data", "bins")
FIG_DIR = os.path.join(JB_DIR, "paper", "figure")
