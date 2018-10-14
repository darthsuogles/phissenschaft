import os
from pathlib import Path
import subprocess

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from tools.hack import *
_THIS_MODULE = git_repo_root() / 'dlsys'

add_to_path(_THIS_MODULE / 'detectron')
