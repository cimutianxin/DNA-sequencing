import matplotlib.pyplot as plt
import pyabf
import numpy as np


def plot_abf(abf, start_index=None, end_index=None, title_prefix="", sample_rate=500): 
    """
    画abf图，接受100倍降采样后的abf数据
    根据index设置起点和终点，对于5kHz的abf，使用秒数*5000
    """
    if start_index is None:
        start_index = 0
        end_index = len(abf.sweepY)
    start_index = int(start_index)
    end_index = int(end_index)

    reduction_rate = sample_rate*10
    duration = len(abf.sweepY) / reduction_rate / 60
    sweepY = abf.sweepY[start_index:end_index]
    sweepX = abf.sweepX[start_index:end_index]

    fig = plt.figure(figsize=(30, 9))
    plt.title(
        "DNA seqencing %s during %.2fs to %.2fs, total %.2f minutes"
        % (
            title_prefix,
            start_index / reduction_rate,
            end_index / reduction_rate,
            duration,
        )
    )
    plt.ylabel(abf.sweepLabelY)
    plt.xlabel(abf.sweepLabelX)
    plt.step(sweepX, sweepY, alpha=0.75, where="post")
    return fig



def step_abf_lvls(abf, df_lvl, typical=False, title_prefix=""):
    """
    在abf图上画出level的step图
    """
    fig = plt.figure(figsize=(30, 9))
    # 作图范围
    start_index = int(df_lvl.iloc[0]["change_point"])
    end_index = int(df_lvl.iloc[-1]["change_point"])
    if typical:
        start_index = start_index + 2000
        end_index = start_index + 5000
    # label
    plt.title("%s" % title_prefix)
    plt.ylabel(abf.sweepLabelY)
    plt.xlabel(abf.sweepLabelX)
    # 作abf图
    plt.step(
        abf.sweepX[start_index:end_index],
        abf.sweepY[start_index:end_index],
        alpha=0.75,
        where="mid",
    )
    # 作level图
    x = [
        df_lvl.iloc[i]["change_point"] / 5000
        for i in range(len(df_lvl))
        if start_index <= df_lvl.iloc[i]["change_point"] <= end_index
    ]
    y = [
        df_lvl.iloc[i]["level"]
        for i in range(len(df_lvl))
        if start_index <= df_lvl.iloc[i]["change_point"] <= end_index
    ]
    plt.step(x, y, color="r", alpha=0.8, where="post", linewidth=2.0)

    return fig

def step_abf_lvls_250(abf, df_lvl, typical=False, title_prefix=""):
    """
    在abf图上画出level的step图
    """
    fig = plt.figure(figsize=(30, 9))
    # 作图范围
    start_index = int(df_lvl.iloc[0]["change_point"])
    end_index = int(df_lvl.iloc[-1]["change_point"])
    if typical:
        start_index = start_index + 2000
        end_index = start_index + 5000
    # label
    plt.title("%s" % title_prefix)
    plt.ylabel(abf.sweepLabelY)
    plt.xlabel(abf.sweepLabelX)
    # 作abf图
    plt.step(
        abf.sweepX[start_index:end_index],
        abf.sweepY[start_index:end_index],
        alpha=0.75,
        where="mid",
    )
    # 作level图
    x = [
        df_lvl.iloc[i]["change_point"] / 2500
        for i in range(len(df_lvl))
        if start_index <= df_lvl.iloc[i]["change_point"] <= end_index
    ]
    y = [
        df_lvl.iloc[i]["level"]
        for i in range(len(df_lvl))
        if start_index <= df_lvl.iloc[i]["change_point"] <= end_index
    ]
    plt.step(x, y, color="r", alpha=0.8, where="post", linewidth=2.0)

    return fig