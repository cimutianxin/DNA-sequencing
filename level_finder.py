# change point 检测
# 输入整段 abf 序列
# 参数设置为
# 输出分段的阶梯图
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

sys.setrecursionlimit(100000)

# abf_A1 = pyabf.ABF("/media/xjtu/7A68EF0368EEBCCF/DNA sequencing/2024_04_11_0000.abf")


def preprocess(abf, filter=True, SIGMA=0.01, DOWNSP_RATE=100):
    if filter:
        pyabf.filter.gaussian(abf, 0)  # remove old filter
        pyabf.filter.gaussian(abf, SIGMA)  # apply new filter
    abf.setSweep(0)

    if len(abf.sweepY) % DOWNSP_RATE != 0:
        abf.sweepY = abf.sweepY[: -(len(abf.sweepY) % DOWNSP_RATE)]
        abf.sweepX = abf.sweepX[: -(len(abf.sweepX) % DOWNSP_RATE)]

    abf.sweepY = np.mean(abf.sweepY.reshape(-1, DOWNSP_RATE), axis=1)
    abf.sweepX = np.mean(abf.sweepX.reshape(-1, DOWNSP_RATE), axis=1)
    return abf


def downsample(abf, DOWNSP_RATE=100):
    abf.setSweep(0)
    if len(abf.sweepY) % DOWNSP_RATE != 0:
        abf.sweepY = abf.sweepY[: -(len(abf.sweepY) % DOWNSP_RATE)]
        abf.sweepX = abf.sweepX[: -(len(abf.sweepX) % DOWNSP_RATE)]

    abf.sweepY = np.mean(abf.sweepY.reshape(-1, DOWNSP_RATE), axis=1)
    abf.sweepX = abf.sweepX.reshape(-1, DOWNSP_RATE)[:, 0]
    return abf


def extra_process(abf):
    """
    对 abf 数据进行预处理
    1. event 内的中值
    2. 除去中值偏差+-3std的数据
    """
    return abf


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


def find_events_arr(arr):
    """
    找出有效的区间ranges
    [->
    time_ranges: sweepX上的秒数; index_ranges: 起止index
    """
    index_ranges = []

    # 找出电流大小在一定范围内(1,60)的连续区间
    start = None
    for i, value in enumerate(arr):
        if -10 <= value <= 80:
            if start is None:
                start = i
        else:
            if start is not None:
                index_ranges.append((start, i))
                start = None
    if start is not None:
        index_ranges.append((start, len(arr) - 1))

    # 除去太短的部分 1 s = 454 个碱基 = 5000 index
    # 原文：discard events of duration < 1 second
    index_ranges = [(start, end) for start, end in index_ranges if end - start > 1000]

    # 输出sweepX的ranges部分
    time_ranges = []
    for start, end in index_ranges:
        time_ranges.append((start / 5000, end / 5000))

    return index_ranges, time_ranges


def find_events(abf):
    """
    找出有效的区间ranges
    [->
    time_ranges: sweepX上的秒数; index_ranges: 起止index
    TODO：在find_change_points, 大于10s的区间效率不高, 故将大于10s的区间分割
    """
    index_ranges = []

    sweepY = np.array(abf.sweepY)
    sweepX = np.array(abf.sweepX)

    # 找出电流大小在一定范围内(1,60)的连续区间
    start = None
    for i, value in enumerate(sweepY):
        if 1 <= value <= 80:
            if start is None:
                start = i
        else:
            if start is not None:
                index_ranges.append((start, i))
                start = None
    if start is not None:
        index_ranges.append((start, len(sweepY) - 1))

    # 除去太短的部分 1 s = 454 个碱基 = 5000 index
    # 原文：discard events of duration < 1 second
    index_ranges = [(start, end) for start, end in index_ranges if end - start > 5000]

    # 如果一个区间内的数据点的标准差小于某个阈值，那么这个区间就是一个有效的区间，A1的这一段是3.2497375，论文设置为5
    # max_std = 10
    # index_ranges = [(start, end) for start, end in index_ranges if np.std(sweepY[start:end]) < max_std]

    # 如果一个区间大于10s，那么将其分割
    # for start, end in index_ranges:
    #     if end - start > 50000:
    #         index_ranges.remove((start, end))
    #         index_ranges.append((start, start + 100000))
    #         index_ranges.append((start + 100000, end))

    # 输出sweepX的ranges部分
    time_ranges = []
    for start, end in index_ranges:
        time_ranges.append((sweepX[start], sweepX[end]))

    return index_ranges, time_ranges


def change_points(abf, ranges) -> list:
    """
    ranges:是一个list，每个元素是一个tuple，表示一个有效区间的起止index
    在每个有效区间内寻找change points
    当一个区间大于10s时，其效果不好
    """
    change_points = []
    for start, end in ranges:
        y = np.array(abf.sweepY[start:end])
        change_points.append([i + start for i in find_change_points(y)])
    return change_points


def find_change_points(y, threshold=-50, mindur=5) -> list:
    """
    产生最小 logp(t1, t2, t3) 的 t2 指示 t1 和 t3 之间当前观测值内可能的电平转换的位置。
    在我们的水平查找算法中，我们从给定的时间窗口 ([t1 ,t3]) 开始，并搜索使 logp 最小化的 t2。
    如果 min(logp) 小于指定阈值（我们使用的阈值 logp = -50), 则在 t2 处存在级别转换，
    并且我们在 t1 和 t2 之间以及 t2 和 t3 之间递归搜索其他级别转换。
    如果 min(logp) 高于原始时间窗口的阈值，则 t1 和 t3 内没有转换，
    我们通过增加 t3 的值来考虑更大的窗口。
    其中
    log p(t1,t2,t3) = (t2-t1) log std(t2,t1)
                    + (t3-t2) log std(t3,t2)
                    + (t3-t1) log std(t3,t1)
                    + const
    mindur: 最小持续时间, 假设change point之间的最小距离
    """

    def logp(t1, t2, t3):
        return -(
            (t2 - t1) * np.log(np.std(y[t1:t2]))
            + (t3 - t2) * np.log(np.std(y[t2:t3]))
            + (t3 - t1) * np.log(np.std(y[t1:t3]))
        )

    def search(t1, t3):
        min_logp = inf
        min_t2 = None
        for t2 in range(t1 + mindur, t3 - mindur + 1):
            current_logp = logp(t1, t2, t3)
            if current_logp < min_logp:
                min_logp = current_logp
                min_t2 = t2
        return min_t2, min_logp

    def find(t1, t3):
        # if t3 >= len(y):
        #     return []
        # print("searching:", t1, t3)
        if t3 - t1 < mindur * 2:  # 原文没出现 mindur 的数值
            return []
        t2, min_logp = search(t1, t3)
        if min_logp < threshold:
            #    print("found:", t2)
            return find(t1, t2) + [t2] + find(t2, t3)
        else:
            return []

    return [0] + find(0, len(y) - 1) + [len(y) - 1]

    # change_points = []
    # while t3 < len(y):
    #     t2, min_logp = search(t1, t3)
    #     if min_logp < threshold:
    #         change_points.append(t2)
    #         t1 = t2
    #         t3 = t1 + 2
    #     else:
    #         t3 += 1
    # return change_points


def calc_level(abf, sub_change_points) -> pd.DataFrame:
    """
    index
    change_point
    level: i和i+1之间的平均值
    """
    df = pd.DataFrame(columns=["change_point", "level"])
    df = df.astype({"change_point": int, "level": float})
    for i in range(len(sub_change_points) - 1):
        df.loc[len(df)] = {
            "change_point": sub_change_points[i],
            "level": np.mean(abf.sweepY[sub_change_points[i] : sub_change_points[i + 1]]),
        }
    df.loc[len(df)] = {"change_point": sub_change_points[-1], "level": df.iloc[-1]["level"]}  # 补全终点
    return df


def calc_level4arr(arr, sub_change_points) -> pd.DataFrame:
    """
    index
    change_point
    level: i和i+1之间的平均值
    """
    df = pd.DataFrame(columns=["change_point", "level"])
    df = df.astype({"change_point": int, "level": float})
    for i in range(len(sub_change_points) - 1):
        df.loc[len(df)] = {
            "change_point": sub_change_points[i],
            "level": np.mean(arr[sub_change_points[i] : sub_change_points[i + 1]]),
        }
    df.loc[len(df)] = {"change_point": sub_change_points[-1], "level": df.iloc[-1]["level"]}  # 补全终点
    return df


# abf_A1 = downsample(abf_A1, 100)
# i_ranges_A1, t_ranges_A1 = find_events(abf_A1)
# print("A1 时间范围：", t_ranges_A1, "index范围: ", i_ranges_A1)
# change_points_A1 = change_points(abf_A1, i_ranges_A1)


def plot_autocorrelation(arr, lags=1000):
    """
    计算并绘制自相关系数
    :param arr: 输入的时间序列数据
    :param lags: 滞后阶数a, 默认为1000即0.2s
    :return: None
    """
    plt.figure(figsize=(12, 6))
    plot_acf(arr, lags=lags)
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Autocorrelation Function")
    plt.show()
