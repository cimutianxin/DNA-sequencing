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


def abf2arr(abf):
    return abf.sweepY


def replace_outliers_with_bounds(arr, window_size=100):
    # 创建滑动窗口视图
    windows = sliding_window_view(arr, window_size)
    half_window = window_size // 2

    result = arr.copy()
    for i in range(half_window, len(arr) - half_window):
        window = windows[i - half_window]
        median = np.median(window)
        std_dev = np.std(window)
        lower_bound = median - 3 * std_dev
        upper_bound = median + 3 * std_dev

        if result[i] < lower_bound:
            result[i] = lower_bound
        elif result[i] > upper_bound:
            result[i] = upper_bound

    return result
