#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 06:45:56 2023

@author: joern
"""

# %% imports

import os

os.chdir("/home/joern/Aktuell/GenerativeESOM/08AnalyseProgramme/Python/")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd



# %% Funtions
def annotate_axes(ax, text, fontsize=18):
    ax.text(-.03, 1.03, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="black")


def annotate_axes_3d(ax, text, fontsize=18):
    ax.text(203, 203, 103, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=fontsize, color="black")

color_pal = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

# %% Paths
pfad_o = "/home/joern/Aktuell/GenerativeESOM/"
pfad_u1 = "08AnalyseProgramme/R/"

# %% Plot selteced FCPS data set in 3D

Dataset_to_plot = "Atom"

with sns.axes_style("whitegrid"):
    fig = plt.figure(figsize=(9, 8))
    gs0 = gridspec.GridSpec(1, 1, figure=fig)

    Data = pd.read_csv(pfad_o + pfad_u1 + Dataset_to_plot + "_OriginalData.csv", index_col=0)

    # Create a mapping from unique classes to colors
    unique_classes = sorted(Data["Cls"].unique())
    class_to_color = {cls: color_pal[i] for i, cls in enumerate(unique_classes)}

    # Map each data point's class to corresponding color
    colors = Data["Cls"].map(class_to_color)

    if len(Data.columns) > 3:
        ax1 = fig.add_subplot(gs0[0, 0], projection='3d')
        scatter = ax1.scatter(
            xs=Data["X1"], ys=Data["X2"], zs=Data["X3"],
            c=colors
        )
        ax1.set(xlabel="x", ylabel="y", zlabel="z")
        ax1.view_init(30, 320)
    else:
        ax1 = fig.add_subplot(gs0[0, 0])
        scatter = ax1.scatter(
            xs=Data["X1"], ys=Data["X2"],
            c=colors,
            alpha=1
        )
        ax1.set(xlabel="x", ylabel="y")

    ax1.set_title(Dataset_to_plot, loc='left')

plt.show()
fig.savefig(Dataset_to_plot + ".svg", format="svg", bbox_inches='tight')


