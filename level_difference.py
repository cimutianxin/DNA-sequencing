import pyabf
import pyabf.filter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pandas as pd
import sys
from math import inf
from multiprocessing import Pool, Manager
from jax import jit
from functools import partial
import jax.numpy as jnp
from numpy.lib.stride_tricks import sliding_window_view
from statsmodels.graphics.tsaplots import plot_acf


def level_difference(arr):
    # 作二维差异图
    for i in range(1, len(arr)):
        diff = abs(arr[i] - arr[i - 1])
