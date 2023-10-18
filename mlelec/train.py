# Overall train script - assembles required pipeline - trains/loads and saves model
import sys
import os
import torch
from data.dataset import Dataset


# ---data---
# attempt to read data ( structures, targets)
# if not found, calculate them with pyscf
# generate a dataset object

# ---model---
###---features--- (if necessary)
# attempt to read features if required for prescribed model
# calculate them if necessary

# instantiate model
# Check for a checkpoint

# Train


# save model
