import os
import torch
import pandas as pd
from pathlib import Path
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

fp_data_root = Path.home() / 'local' / 'data'
fp_face_dataset = fp_data_root / 'faces'
assert fp_face_dataset.exists(), \
    "Please download the dataset: https://download.pytorch.org/tutorial/faces.zip"

df_landmarks = pd.read_csv(str(fp_face_dataset / "face_landmarks.csv"))
