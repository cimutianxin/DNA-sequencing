import matplotlib.pyplot as plt
import pyabf
import numpy as np


def plot_abf(abf, start_index=None, end_index=None, title_prefix=""):
    reduction_rate = 5000
    if end_index is None:
        fig = plt.figure(figsize=(30, 9))
        duration = len(abf.sweepY) / reduction_rate / 60
        sweepY = abf.sweepY
        sweepX = abf.sweepX
        plt.title("DNA seqencing %s, total %.2f minutes" % (title_prefix, duration))
        plt.ylabel(abf.sweepLabelY)
        plt.xlabel(abf.sweepLabelX)
        plt.step(sweepX, sweepY, alpha=0.75, where="post")
        return fig
    else:
        start_index = int(start_index * reduction_rate)
        end_index = int(end_index * reduction_rate)
        duration = len(abf.sweepY) / reduction_rate / 60
        fig = plt.figure(figsize=(30, 9))
        sweepY = abf.sweepY[start_index:end_index]
        sweepX = abf.sweepX[start_index:end_index]
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
