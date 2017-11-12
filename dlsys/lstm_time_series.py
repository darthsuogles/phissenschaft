import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from datasets import series

df = series.beijing_pollution()

df[['pm2.5', 'TEMP', 'PRES']].plot()
