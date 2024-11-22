# -*- coding: utf-8 -*-

from random import randint

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from cycler import cycler
import seaborn as sns
from scipy import stats

import rwth_colors

cm = 1 / 2.54  # centimeters in inches

# column-name -> metric unit
metric_dic = {
    "AH_throughput": "Ah",
    "Temperature": "$^\circ$C",
    "Voltage": "V",
    "Capacity": "Ah",
    "Duration": "days",
    "SOC": "$\%$",
    "SOH": "$\%$",
    "Current": "A",
    "Wh_throughput": "Wh",
    "Capacity_current": "A",
    "Z_phase": "rad",
    "Z_abs": "$\Omega$",
}

# if no color is selected, use the default colors of the cycler, which are:
rwth_colors_cycler_color = [
    rwth_colors.colors[("blue", 100)],
    rwth_colors.colors[("black", 100)],
    rwth_colors.colors[("magenta", 100)],
    rwth_colors.colors[("yellow", 100)],
    rwth_colors.colors[("green", 100)],
    rwth_colors.colors[("bordeaux", 100)],
    rwth_colors.colors[("orange", 100)],
    rwth_colors.colors[("turqoise", 100)],
    rwth_colors.colors[("darkred", 100)],
    rwth_colors.colors[("lime", 100)],
    rwth_colors.colors[("petrol", 100)],
    rwth_colors.colors[("lavender", 100)],
    rwth_colors.colors[("red", 100)],
    rwth_colors.colors[("blue", 50)],
    rwth_colors.colors[("black", 50)],
    rwth_colors.colors[("magenta", 50)],
    rwth_colors.colors[("yellow", 50)],
    rwth_colors.colors[("green", 50)],
    rwth_colors.colors[("bordeaux", 50)],
    rwth_colors.colors[("orange", 50)],
    rwth_colors.colors[("turqoise", 50)],
    rwth_colors.colors[("darkred", 50)],
    rwth_colors.colors[("lime", 50)],
    rwth_colors.colors[("petrol", 50)],
    rwth_colors.colors[("lavender", 50)],
    rwth_colors.colors[("red", 50)],
]

rwth_colors_cycler_linestyle = ["-", "--", "-."]


cc = cycler(linestyle=rwth_colors_cycler_linestyle) * cycler(
    color=rwth_colors_cycler_color
)

sns.set(style="white")

mpl.rcParams["axes.prop_cycle"] = cc
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["image.cmap"] = "turbo"
font_size = 8
mpl.rcParams["axes.labelsize"] = font_size
mpl.rcParams["axes.titlesize"] = font_size
mpl.rcParams["font.size"] = font_size
mpl.rcParams["xtick.labelsize"] = font_size
mpl.rcParams["ytick.labelsize"] = font_size
mpl.rcParams["legend.fontsize"] = font_size


def get_single_cellnames(cell_list, str_cutoff=-11):
    """
    Given a list of cell names, return a list of the unique cell names.

    Parameters:
        cell_list (list): The list of cell names.
        str_cutoff (int): The number of characters to cut off the end of the cell name.

    Returns:
        list: The list of unique cell names.
    """
    cell_names = [filename[:str_cutoff] for filename in cell_list]
    single_cellnames = list(set(cell_names))
    return single_cellnames


def setup_scatter(
    dataset,
    rmse,
    title=True,
    legend=False,
    fig=None,
    ax=None,
    ax_xlabel=True,
    ax_ylabel=True,
    subplots_adjust=True,
    add_trendline=True,
    label=None,
):
    """
    Setup the scatter plot for the given dataset.

    Parameters:
        dataset (dict): The dataset dictionary containing the dataframes and labels.
        rmse (float): The root mean squared error of the model.
        title (bool): Whether or not to show the title.
        legend (bool): Whether or not to show the legend.
        fig (Figure): The figure to plot on.
        ax (Axes): The axis to plot on.
        ax_xlabel (bool): Whether or not to show the x-axis label.
        ax_ylabel (bool): Whether or not to show the y-axis label.
        subplots_adjust (bool): Whether or not to adjust the subplots.
        add_trendline (bool): Whether or not to add a trendline to the plot.

    Returns:
        Tuple[Figure, Axes]: The figure and axis objects.
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)
    ax.grid(True)
    ax.set_aspect("equal", "box")

    if label is None:
        label = dataset["label"]
        label = "".join(label)
    label_latex = label
    if mpl.rcParams["text.usetex"]:
        label_latex = label_latex.replace("_", "\\textunderscore ")

    if ax_xlabel:
        ax.set_xlabel("actual: " + label_latex + " in " + metric_dic[label])
    if ax_ylabel:
        ax.set_ylabel("predicted: " + label_latex + " in " + metric_dic[label])

    if legend:
        ax.legend()

    if add_trendline:
        y_train = dataset["train"][1]
        # values for line plot in range of test data
        x = np.linspace(y_train.min(), y_train.max(), num=4)

        # plot ideal line
        ax.plot(x, x, label="ideal", color=rwth_colors.colors[("green", 100)])

    if title:
        title = str("Prediction of " + label_latex + " (rmse=" + str(rmse)[:4] + ")")
        ax.set_title(title)

    if subplots_adjust:
        plt.subplots_adjust(bottom=0.14)

    return fig, ax


def get_cell_name_index(df, cell_name):
    """
    Given a dataframe and a cell name, return the index of the cell name in the dataframe.

    Parameters:
        df (DataFrame): The dataframe we are working with.
        cell_name (str): The name of the cell we are looking for.

    Returns:
        Index: The index of the cell name in the dataframe.
    """
    return df.index.get_level_values(0).str.contains(cell_name)


def get_cell_label(dic, cell_name):
    """
    Given a dictionary and a cell name, return the label for that cell.

    Parameters:
        dic (dict): The dictionary of cell labels.
        cell_name (str): The name of the cell.

    Returns:
        str: The label for the cell.
    """
    if dic:
        cell_label = dic[cell_name]
    else:
        cell_label = cell_name
    return cell_label


def cell_scatter(
    dataset,
    Y_predicted,
    is_test=False,
    is_validation=False,
    cell_names=[""],
    dic=None,
    title=True,
    legend=False,
    fig=None,
    ax=None,
    ax_xlabel=True,
    ax_ylabel=True,
    subplots_adjust=True,
    add_trendline=True,
):
    """
    Plot the predicted values for each cell in the dataset.

    Parameters:
        dataset (dict): The dataset dictionary containing the dataframes and labels.
        Y_predicted (array-like): The predicted values for each cell.
        is_train (bool): If True, the training data is plotted.
        cell_names (list): The cell names to plot.
        dic (dict): The dictionary of cell names and labels.
        title (bool): Whether or not to show the title of the plot.
        legend (bool): Whether or not to show the legend.
        fig: The figure to plot on.
        ax: The axis to plot on.
        ax_xlabel (bool): Whether or not to show the x-axis label.
        ax_ylabel (bool): Whether or not to show the y-axis label.
        subplots_adjust (bool): Whether or not to adjust the subplots.
        add_trendline (bool): Whether or not to add a trendline to the plot.

    Returns:
        fig: The figure.
        ax: The axis.
    """
    # get label/target string and dataframe of test data
    if is_test:
        df_plot = dataset["df_test"]
        scatter_color = mpl.colors.to_rgba(rwth_colors.colors[("blue", 100)], 0.5)
        scatter_marker = "2"
    elif is_validation:
        df_plot = dataset["df_validation"]
        scatter_color = mpl.colors.to_rgba(rwth_colors.colors[("turqoise", 100)], 0.5)
        scatter_marker = "1"
    else:
        df_plot = dataset["df_train"]
        scatter_color = mpl.colors.to_rgba(rwth_colors.colors[("petrol", 100)], 0.5)
        scatter_marker = "."

    label = dataset["label"]

    # calculate rmse
    Y_test = df_plot[label].to_numpy().ravel()
    Y_predicted_copy = Y_predicted.copy()
    Y_predicted_copy = Y_predicted_copy.ravel()
    rmse = np.sqrt(((Y_predicted_copy - Y_test) ** 2).mean())

    fig, ax = setup_scatter(
        dataset,
        rmse,
        title=title,
        legend=legend,
        fig=fig,
        ax=ax,
        ax_xlabel=ax_xlabel,
        ax_ylabel=ax_ylabel,
        subplots_adjust=subplots_adjust,
        add_trendline=add_trendline,
    )

    for i, cell_name in enumerate(cell_names):
        # select subset of df with all cells having cell_name in their cell_ID
        indexes = get_cell_name_index(df_plot, cell_name)
        df_cell_name = df_plot.loc[indexes]

        # select test ys with cell_name
        y_test_cell_name = df_cell_name[label].to_numpy()

        # select corresponding predictions
        Y_predicted_ser = pd.Series(Y_predicted_copy)
        Y_predicted_cell_name = Y_predicted_ser.loc[indexes].to_numpy()

        # get label for the legend
        scatter_label = get_cell_label(dic, cell_name)

        ax.plot(
            y_test_cell_name,
            Y_predicted_cell_name,
            linestyle="",
            label=scatter_label,
            marker=scatter_marker,
            color=scatter_color,
            mfc=scatter_color,
        )

    if legend:
        ax.legend()

    return fig, ax


def cell_scatter_nn(dataset, Y_predicted, **kwargs):
    """
    Plot the predicted values for each cell in the dataset. Difference to cell_scatter is that it can deal with multidimensional labels

    Parameters:
        dataset (dict): The dataset dictionary containing the dataframes and labels.
        Y_predicted (array-like): The predicted values for each cell.
        is_train (bool): If True, the training data is plotted.
        cell_names (list): The cell names to plot.
        dic (dict): The dictionary of cell names and labels.
        title (bool): Whether or not to show the title of the plot.
        legend (bool): Whether or not to show the legend.
        fig: The figure to plot on.
        ax: The axis to plot on.
        ax_xlabel (bool): Whether or not to show the x-axis label.
        ax_ylabel (bool): Whether or not to show the y-axis label.
        subplots_adjust (bool): Whether or not to adjust the subplots.
        add_trendline (bool): Whether or not to add a trendline to the plot.

    Returns:
        fig: The figure.
        ax: The axis.
    """
    label_origin = dataset["label"]

    if type(label_origin) is str:
        label = [label_origin]
    else:
        label = label_origin

    fig, axes = plt.subplots(1, len(label))
    for label_index, single_label in enumerate(label):
        if len(label) == 1:
            ax = axes
        else:
            ax = axes[label_index]
        dataset["label"] = single_label
        y_pred = Y_predicted[:, label_index]
        cell_scatter(dataset, y_pred, fig=fig, ax=ax, **kwargs)
    dataset["label"] = label_origin
    return fig, ax


def reduce_df(df, feature, nr_intervals, interval_type="lin"):
    """
    Reduce the dataframe to a smaller dataframe containing only the data for the given feature.

    Parameters:
        df (DataFrame): The dataframe containing the data.
        feature (str): The feature we are reducing.
        nr_intervals (int): The number of intervals we are reducing to.
        interval_type (str): Type of the intervals, 'lin' or 'log'.

    Returns:
        DataFrame: The reduced dataframe.
    """
    if interval_type not in ["lin", "log"]:
        interval_type = "lin"

    if interval_type == "lin":
        feature_steps = np.linspace(
            np.min(df[feature]), np.max(df[feature]), nr_intervals + 1
        )
    else:
        feature_steps = np.logspace(
            np.log10(np.min(df[feature])),
            np.log10(np.max(df[feature])),
            nr_intervals + 1,
        )

    df_new = pd.DataFrame()
    for i, step in enumerate(feature_steps[:-1]):
        if i == 0:
            mask = (df[feature] >= feature_steps[i]) & (
                df[feature] <= feature_steps[i + 1]
            )
            possible = df.loc[mask]
        else:
            mask = (df[feature] > feature_steps[i]) & (
                df[feature] <= feature_steps[i + 1]
            )
            possible = df.loc[mask]
        if len(possible) > 0:
            row = possible.iloc[[int(np.floor(len(possible) / 2))]]
            df_new = pd.concat([df_new, row])
    df_new = df_new[~df_new.duplicated(keep="first")]
    return df_new


def setup_Bode(
    title=None,
    legend=False,
    fig=None,
    ax1=None,
    ax2=None,
    ax1_xlabel=False,
    ax1_ylabel=True,
    ax2_xlabel=True,
    ax2_ylabel=True,
    subplots_adjust=True,
):
    """
    Set up the Bode plot.

    Parameters:
        title (str): The title of the plot.
        legend (bool): Whether or not to show the legend.
        fig: The figure of the plot.
        ax1: The first axis of the plot.
        ax2: The second axis of the plot.
        ax1_xlabel (bool): Whether or not to show the xlabel of the first axis.
        ax1_ylabel (bool): Whether or not to show the ylabel of the first axis.
        ax2_xlabel (bool): Whether or not to show the xlabel of the second axis.
        ax2_ylabel (bool): Whether or not to show the ylabel of the second axis.
        subplots_adjust (bool): Whether or not to adjust the subplots.

    Returns:
        fig: The figure.
        ax1: The first axis.
        ax2: The second axis.
    """
    if fig is None or ax1 is None or ax2 is None:
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, ncols=1, sharex=True, figsize=(20 * cm, 9 * cm)
        )
        if ax2_xlabel:
            ax2.set_xlabel(r"Frequency in Hz")
    else:
        if ax1_xlabel:
            ax1.set_xlabel(r"Frequency in Hz")
        if ax2_xlabel:
            ax2.set_xlabel(r"Frequency in Hz")
    ax2.grid(True, which="major", axis="both", alpha=1)
    ax2.grid(True, which="minor", axis="x", alpha=0.2)
    ax2.set_xscale("log")
    if ax2_ylabel:
        if mpl.rcParams["text.usetex"]:
            ax2.set_ylabel(r"\ \angle $\underline{Z}$ in $^\circ$")
        else:
            ax2.set_ylabel(r"Phase of Z in °")

    ax1.grid(True, which="major", axis="both", alpha=1)
    ax1.grid(True, which="minor", axis="x", alpha=0.2)
    ax1.set_xscale("log")
    if ax1_ylabel:
        if mpl.rcParams["text.usetex"]:
            ax1.set_ylabel(r"$|\underline{Z}|$ in m$\Omega$")
        else:
            ax1.set_ylabel(r"|Z| in mΩ")

    fig.suptitle(title)

    if legend:
        ax1.legend()
    if subplots_adjust:
        plt.subplots_adjust(bottom=0.14)

    return fig, (ax1, ax2)


def setup_Nyq(
    title=None,
    legend=False,
    fig=None,
    ax=None,
    ax_xlabel=True,
    ax_ylabel=True,
    subplots_adjust=True,
):
    """
    Set up the Nyquist plot.

    Parameters:
        title (str): The title of the plot.
        legend (bool): Whether or not to show the legend.
        fig: The figure to plot on.
        ax: The axis to plot on.
        ax_xlabel (bool): Whether or not to show the x-axis label.
        ax_ylabel (bool): Whether or not to show the y-axis label.
        subplots_adjust (bool): Whether or not to adjust the subplots.

    Returns:
        fig: The figure.
        ax: The axis.
    """

    if fig is None or ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20 * cm, 9 * cm))
    ax.grid(True)
    ax.set_aspect("equal", "box")
    if ax_xlabel:
        if mpl.rcParams["text.usetex"]:
            ax.set_xlabel(r"$\Re(\underline{Z})$ in m$\Omega$")
        else:
            ax.set_xlabel(r"Re(Z) in mΩ")
    if ax_ylabel:
        if mpl.rcParams["text.usetex"]:
            ax.set_ylabel(r"$\Im(\underline{Z})$ in m$\Omega$")
        else:
            ax.set_ylabel(r"Im(Z) in mΩ")
    fig.suptitle(title)

    if legend:
        ax.legend()

    if subplots_adjust:
        plt.subplots_adjust(bottom=0.14)

    if not ax.yaxis_inverted():
        ax.invert_yaxis()

    return fig, ax


def setup_colormap(min_value, max_value, feature, fig, axes, location="right"):
    """
    Set up the color map for the plots.

    Parameters:
        min_value (float): The minimum value of the color map.
        max_value (float): The maximum value of the color map.
        feature (str): The feature being plotted.
        fig: The figure to add the color bar to.
        axes: The axes to add the color bar to.
        location (str): The location of the color bar.

    Returns:
        cmap: The color map.
    """

    norm = mpl.colors.Normalize(min_value, max_value)
    if feature in ["SOH", "SOC", "Voltage", "Capacity"]:
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.turbo_r)
    else:
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.turbo)
    cmap.set_array([])

    cbar = fig.colorbar(cmap, ax=axes, location=location)

    if feature in metric_dic:
        cbar.set_label(feature + " in " + metric_dic[feature])
    else:
        cbar.set_label(feature)

    return cmap, cbar


def plot_bode_feature(
    df,
    key_lookup_df,
    feature,
    title=None,
    reduce=False,
    nr_intervals=10,
    interval_type="lin",
    highlight_freqs=[],
    highlight_df_columns=[],
    fig=None,
    ax1=None,
    ax2=None,
    ax1_xlabel=True,
    ax1_ylabel=True,
    ax2_xlabel=True,
    ax2_ylabel=True,
    subplots_adjust=True,
    cmap=None,
    cbar=None,
    set_limits=True,
    set_ticks=True,
):
    """
    Plot the Bode diagram for a given feature.

    Parameters:
        df: The dataframe containing the data for the feature.
        key_lookup_df: The dataframe containing the key lookup for the feature.
        feature: The feature we are plotting.
        title: The title of the plot.
        reduce: Whether or not to reduce the dataframe to a smaller number of intervals.
        nr_intervals: The number of intervals to reduce to.
        interval_type: The type of intervals to use (linear or logarithmic).
        highlight_freqs: Frequencies to highlight in the plot.
        highlight_df_columns: Columns to highlight in the plot.
        fig: The figure to plot on.
        ax1: The axis to plot the magnitude on.
        ax2: The axis to plot the phase on.
        ax1_xlabel: Whether to show the x-axis label for the magnitude plot.
        ax1_ylabel: Whether to show the y-axis label for the magnitude plot.
        ax2_xlabel: Whether to show the x-axis label for the phase plot.
        ax2_ylabel: Whether to show the y-axis label for the phase plot.
        subplots_adjust: Whether to adjust the subplots layout.
        cmap: The colormap to use for coloring the plot.
        cbar: The color bar to use for coloring the plot.
        set_limits: Whether to set the limits of the plot.
        set_ticks: Whether to set the ticks of the plot.

    Returns:
        fig: The modified figure.
        (ax1, ax2): The modified axes.
        cmap: The modified colormap.
        cbar: The modified color bar.
    """
    phase_exists = len(list(filter(lambda x: "EIS_Z_phase" in x, df.columns))) > 0

    if reduce:
        df = reduce_df(df, feature, nr_intervals, interval_type=interval_type)

    # setup figure and colormap
    if fig is None or ax1 is None or ax2 is None:
        fig, (ax1, ax2) = setup_Bode(title)
    else:
        fig, (ax1, ax2) = setup_Bode(
            title,
            fig=fig,
            ax1=ax1,
            ax2=ax2,
            ax1_xlabel=ax1_xlabel,
            ax1_ylabel=ax1_ylabel,
            ax2_xlabel=ax2_xlabel,
            ax2_ylabel=ax2_ylabel,
            subplots_adjust=subplots_adjust,
        )

    if cmap is None or cbar is None:
        cmap, cbar = setup_colormap(
            df[feature].min(), df[feature].max(), feature, fig, (ax1, ax2)
        )

    # get frequencies and keys of dataframe EIS columns
    frequencies = key_lookup_df["frequency"].to_numpy()
    abs_keys = key_lookup_df["EIS_Z_abs"].to_list()
    if phase_exists:
        phase_keys = key_lookup_df["EIS_Z_phase"].to_list()

    segs_abs = np.zeros((len(df[feature]), len(abs_keys), 2))
    segs_abs[:, :, 0] = frequencies
    segs_abs[:, :, 1] = df[abs_keys] * 1000

    if phase_exists:
        segs_phase = np.zeros((len(df[feature]), len(phase_keys), 2))
        segs_phase[:, :, 0] = frequencies
        segs_phase[:, :, 1] = df[phase_keys] / np.pi * 180

    if set_limits:
        ax1.set_xlim(segs_abs[:, :, 0].min(), segs_abs[:, :, 0].max())
        ax1.set_ylim(segs_abs[:, :, 1].min(), segs_abs[:, :, 1].max())

        if phase_exists:
            ax2.set_xlim(segs_phase[:, :, 0].min(), segs_phase[:, :, 0].max())
            ax2.set_ylim(segs_phase[:, :, 1].min(), segs_phase[:, :, 1].max())

    # color for both the same
    colors = cmap.to_rgba(df[feature].to_numpy("float64"))

    if highlight_freqs or highlight_df_columns:
        line_segments_abs = mpl.collections.LineCollection(
            segs_abs, colors=colors, linestyle="solid", alpha=0.1
        )
        if phase_exists:
            line_segments_phase = mpl.collections.LineCollection(
                segs_phase, colors=colors, linestyle="solid", alpha=0.1
            )

        ax1.add_collection(line_segments_abs)
        if phase_exists:
            ax2.add_collection(line_segments_phase)
    else:
        line_segments_abs = mpl.collections.LineCollection(
            segs_abs, colors=colors, linestyle="solid", alpha=1
        )
        if phase_exists:
            line_segments_phase = mpl.collections.LineCollection(
                segs_phase, colors=colors, linestyle="solid", alpha=1
            )

        ax1.add_collection(line_segments_abs)
        if phase_exists:
            ax2.add_collection(line_segments_phase)

    if highlight_freqs:
        abs_high_key, phase_high_key = get_highlight_keys_bode(
            key_lookup_df, highlight_freqs
        )

        abs_high_values = df[abs_high_key].to_numpy(dtype="float64") * 1000
        phase_high_values = df[phase_high_key].to_numpy(dtype="float64") / np.pi * 180
        feature_values = df[feature].to_numpy(dtype="float64")

        colors = cmap.to_rgba(feature_values)
        colors = np.repeat(colors, len(abs_high_key), axis=0)

        highlight_freqs = np.tile(highlight_freqs, (len(df[abs_high_key]), 1))

        ax1.scatter(
            highlight_freqs, abs_high_values, s=25, marker="x", c=colors, alpha=1
        )
        ax2.scatter(
            highlight_freqs, phase_high_values, s=25, marker="x", c=colors, alpha=1
        )

    if highlight_df_columns:
        feature_values = df[feature].to_numpy(dtype="float64")
        colors = cmap.to_rgba(feature_values)

        for x, y in highlight_df_columns:
            if "Phase" in y:
                ax2.scatter(
                    df[x], df[y] / np.pi * 180, s=25, marker="x", c=colors, alpha=1
                )
            elif "Abs" in y:
                ax1.scatter(df[x], df[y] * 1000, s=25, marker="x", c=colors, alpha=1)

    delta_y1 = np.median(np.diff(ax1.get_yticks()))
    if phase_exists:
        delta_y2 = np.median(np.diff(ax2.get_yticks()))

    y1_min = np.min(ax1.get_yticks())
    y1_max = np.max(ax1.get_yticks())

    if phase_exists:
        y2_min = np.min(ax2.get_yticks())
        y2_max = np.max(ax2.get_yticks())

    if set_ticks:
        # always go for the delta_x, there the font size limits everything
        ax1.set_yticks(np.arange(y1_min, y1_max + delta_y1, delta_y1))
        if phase_exists:
            ax2.set_yticks(np.arange(y2_min, y2_max + delta_y2, delta_y2))

    return fig, (ax1, ax2), cmap, cbar


def get_highlight_keys_nyquist(key_lookup_df, highlight_freqs):
    """
    Given a dataframe of key lookup values and a list of frequencies, return the keys that match those frequencies.

    Parameters:
        key_lookup_df: The dataframe of key lookup values.
        highlight_freqs: The list of frequencies.

    Returns:
        The keys that match those frequencies.
    """
    lookup_high = key_lookup_df.loc[key_lookup_df["frequency"].isin(highlight_freqs)]
    return (lookup_high["EIS_Z_Re"].to_list(), lookup_high["EIS_Z_Im"].to_list())


def get_highlight_keys_bode(key_lookup_df, highlight_freqs):
    """
    Given a dataframe of key lookup values and a list of frequencies, return the keys that match those frequencies.

    Parameters:
        key_lookup_df: The dataframe of key lookup values.
        highlight_freqs: The list of frequencies.

    Returns:
        The keys that match those frequencies.
    """
    lookup_high = key_lookup_df.loc[key_lookup_df["frequency"].isin(highlight_freqs)]
    return (lookup_high["EIS_Z_abs"].to_list(), lookup_high["EIS_Z_phase"].to_list())


def plot_nyquist_feature(
    df,
    key_lookup_df,
    feature,
    title=None,
    reduce=False,
    interval_type="lin",
    nr_intervals=10,
    highlight_freqs=[],
    highlight_df_columns=[],
    fig=None,
    ax=None,
    ax_xlabel=True,
    ax_ylabel=True,
    subplots_adjust=True,
    legend=False,
    cmap=None,
    cbar=None,
    set_limits=True,
    set_ticks=True,
):
    """
    Plot the Nyquist diagram for a given feature.

    Parameters:
        df: The dataframe containing the feature data.
        key_lookup_df: The dataframe containing the key lookup data.
        feature: The feature to plot.
        title: The title of the plot.
        reduce: Whether to reduce the dataframe to a single value per feature.
        interval_type: The type of intervals to use when reducing the dataframe.
        nr_intervals: The number of intervals to use when reducing the dataframe.
        highlight_freqs: The frequencies to highlight.
        highlight_df_columns: The dataframe columns to highlight.
        fig: The figure to plot on.
        ax: The axis to plot on.
        ax_xlabel: Whether to show the x-axis label.
        ax_ylabel: Whether to show the y-axis label.
        subplots_adjust: Whether to adjust the subplots.
        legend: Whether to show the legend.
        cmap: The colormap to use.
        cbar: The color bar to use.
        set_limits: Whether to set the limits of the plot.
        set_ticks: Whether to set the ticks of the plot.

    Returns:
        fig: The modified figure.
        ax: The modified axis.
        cmap: The modified colormap.
        cbar: The modified color bar.
    """

    if reduce:
        df = reduce_df(df, feature, nr_intervals, interval_type=interval_type)

    # setup figure and colormap
    if fig is None or ax is None:
        fig, ax = setup_Nyq(title, legend)
    else:
        fig, ax = setup_Nyq(
            title,
            legend,
            fig=fig,
            ax=ax,
            ax_xlabel=ax_xlabel,
            ax_ylabel=ax_ylabel,
            subplots_adjust=subplots_adjust,
        )

    if cmap is None or cbar is None:
        cmap, cbar = setup_colormap(
            df[feature].min(), df[feature].max(), feature, fig, ax
        )
    colors = cmap.to_rgba(df[feature].to_numpy("float64"))

    # get keys of dataframe eis columns
    Re_keys = key_lookup_df["EIS_Z_Re"].to_list()
    Im_keys = key_lookup_df["EIS_Z_Im"].to_list()

    segs_nyq = np.zeros((len(df[feature]), len(Im_keys), 2))
    segs_nyq[:, :, 0] = df[Re_keys] * 1000
    segs_nyq[:, :, 1] = df[Im_keys] * 1000

    if highlight_freqs or highlight_df_columns:
        line_segments_nyq = mpl.collections.LineCollection(
            segs_nyq, colors=colors, linestyle="solid", alpha=0.1
        )
        ax.add_collection(line_segments_nyq)
    else:
        line_segments_nyq = mpl.collections.LineCollection(
            segs_nyq, colors=colors, linestyle="solid", alpha=1
        )
        ax.add_collection(line_segments_nyq)

    if highlight_freqs:
        Re_high_key, Im_high_key = get_highlight_keys_nyquist(
            key_lookup_df, highlight_freqs
        )

        Re_high_values = df[Re_high_key].to_numpy(dtype="float64") * 1000
        Im_high_values = df[Im_high_key].to_numpy(dtype="float64") * 1000
        feature_values = df[feature].to_numpy(dtype="float64")

        colors = cmap.to_rgba(feature_values)
        colors = np.repeat(colors, len(Re_high_key), axis=0)

        ax.scatter(Re_high_values, Im_high_values, s=25, marker="x", c=colors, alpha=1)

    if highlight_df_columns:
        feature_values = df[feature].to_numpy(dtype="float64")
        colors = cmap.to_rgba(feature_values)

        for x, y in highlight_df_columns:
            ax.scatter(df[x] * 1000, df[y] * 1000, s=25, marker="x", c=colors, alpha=1)

    # first based on the data
    x_min = segs_nyq[:, :, 0].min()
    x_max = segs_nyq[:, :, 0].max()

    y_min = segs_nyq[:, :, 1].min()
    y_max = segs_nyq[:, :, 1].max()

    if set_limits:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)

    # now based on the ticks
    delta_x = np.median(np.diff(ax.get_xticks()))
    delta_y = np.median(np.diff(ax.get_yticks()))

    delta = np.max([delta_x, delta_y])

    x_min = np.min(ax.get_xticks())
    x_max = np.max(ax.get_xticks())

    # NYQUIST!!! Y - Axis flipped!!!
    y_min = np.min(ax.get_yticks())
    y_max = np.max(ax.get_yticks())

    if set_ticks:
        # always go for the delta_x, there the font size limits everything
        ax.set_yticks(np.arange(y_min, y_max + delta, delta))
        ax.set_xticks(np.arange(x_min, x_max + delta, delta))

    return fig, ax, cmap, cbar


def cor_matrix(df):
    """
    Plots a correlation matrix for the given data frame.

    Parameters:
        df: The dataframe containing the data.

    Returns:
        A seaborn PairGrid.
    """

    def corr_upper(xdata, ydata, **kwargs):
        nas = np.logical_or(np.isnan(xdata.values), np.isnan(ydata.values))
        cmap = kwargs["cmap"]
        norm = kwargs["norm"]
        ax = plt.gca()
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        sns.despine(ax=ax, bottom=True, top=True, left=True, right=True)
        r, _ = stats.pearsonr(xdata[~nas], ydata[~nas])
        facecolor = cmap(norm(r))
        ax.set_facecolor(facecolor)
        ax.annotate(
            f"r={r:.2f}",
            xy=(0.5, 0.5),
            xycoords=ax.transAxes,
            color="black",
            size=font_size,
            ha="center",
            va="center",
        )

    def set_font(xdata, ydata, **kwargs):
        ax = plt.gca()
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=font_size)
        ax.tick_params(axis="y", labelsize=font_size)

    g = sns.PairGrid(df, dropna=True, diag_sharey=False)
    g.figure.set_size_inches(16 * cm, 16 * cm)

    g.map_upper(
        corr_upper, cmap=plt.get_cmap("turbo"), norm=plt.Normalize(vmin=-1, vmax=1)
    )

    g.map_diag(
        sns.histplot,
        kde=True,
        kde_kws=dict(cut=3),
        alpha=1,
        edgecolor=rwth_colors.colors[("blue", 50)],
        shrink=0.8,
        fill=False,
    )

    g.map_lower(sns.kdeplot, cmap="turbo")
    g.map_lower(sns.rugplot, color=rwth_colors.colors[("blue", 100)])

    g.map(set_font)
    return g


def plot_gradients(gradient_arr, frequencies, ax, title=""):
    """
    plot displays the gradient distribution of multiple models.
    The gradient of the model is the gradient of the output with respect to the first layer of the model.
    It is assumed that the model's first layer is a weakly connected layer with each neuron corresponding to an impedance at a certain frequency.
    The plot shows how the absolute gradient values of different models are distributed.
    The corresponding frequencies of the gradient components are on the logaritmic x axis
    @param gradient_arr - numpy.array containing one 'mean' gradient of each model
    @param frequencies - np.array (1D) containing the frequencies corresponding to the features
    @param title - title of the plot. Makes Sense to adapt it
    @returns a tuple of figure and ax of the plot
    """
    # frequencies = key_lookup_df["frequency"].to_numpy()
    plot_arr = gradient_arr.swapaxes(0, 1)

    # fig, ax = plt.subplots(1, 1)

    # just for the width
    w = 0.1

    def width(p, w):
        return 10 ** (np.log10(p) + w / 2.0) - 10 ** (np.log10(p) - w / 2.0)

    # set properties of plot
    boxprops = dict(linewidth=2.5)
    medianprops = dict(linewidth=3, color="red")

    # just plot
    ax.boxplot(
        list(plot_arr),
        positions=list(frequencies),
        widths=width(frequencies, w),
        boxprops=boxprops,
        medianprops=medianprops,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Frequencies in Hz")
    ax.set_ylabel(
        r"$ \frac{|\nabla_{n_{f}} \hat{y} |}{\Vert \nabla_{n_{f}} \hat{y} \Vert_2 } $",
        rotation="horizontal",
        ha="right",
    )
    ax.set_title(title)
    return ax


def plot_component_gradient(
    grad_arr, key_lookup_df, title="", components=["magnitude", "phase"]
):
    grad_length = grad_arr.shape[1]
    nr_freqs = key_lookup_df.shape[0]
    assert grad_length == 2 * nr_freqs, f" gradient array has length \
        {grad_length} but is has to match \
        the 2 times nr of frequencies in key_lookup_df: {2*nr_freqs}"

    # create 3 subplots in a grid
    fig = plt.figure()
    fig.suptitle(title)
    gs = fig.add_gridspec(2, 2)
    axComponent1 = fig.add_subplot(gs[0, 0])
    axComponent2 = fig.add_subplot(gs[0, 1])
    axSum = fig.add_subplot(gs[1, :])

    comp1_gradients = grad_arr[:, :nr_freqs]
    comp2_gradients = grad_arr[:, nr_freqs:]
    sum_gradients = comp1_gradients + comp2_gradients

    frequencies = key_lookup_df["frequency"].to_numpy()
    plot_gradients(comp1_gradients, frequencies, axComponent1, title=components[0])
    plot_gradients(comp2_gradients, frequencies, axComponent2, title=components[1])
    plot_gradients(sum_gradients, frequencies, axSum, title="sum")

    return fig, (axComponent1, axComponent2, axSum)


def plot_gradient_result(grad_result, title="", include_sum=False):
    feature_keys = grad_result["gradients"].columns.to_list()
    (freq_dic, select_dic) = get_frequency_dic(feature_keys, get_select_dic=True)
    components = list(freq_dic.keys())

    # create 3 subplots in a grid
    fig = plt.figure()
    fig.suptitle(title)

    nr_comp = len(components)
    grad_arr = grad_result["gradients"].to_numpy()

    # map out the layout of the plot
    if include_sum:
        # if the sum of components is included in plot then there needs to be an extra row
        gs = fig.add_gridspec(2, nr_comp)
        grad_len = len(select_dic[components[0]])
        comp_gradients = np.zeros([nr_comp, np.shape(grad_arr)[0], grad_len])

        axes_components = [fig.add_subplot(gs[0, i]) for i in range(nr_comp)]
        axSum = fig.add_subplot(gs[1, :])
        axes_components.append(axSum)
        # TODO include assert to check frequency equality
    else:
        # otherwise just plot beneath
        gs = fig.add_gridspec(nr_comp, 1)
        axes_components = [fig.add_subplot(gs[i, 0]) for i in range(nr_comp)]

    for nr_comp, comp in enumerate(components):
        gradients = grad_arr[:, select_dic[comp]]
        plot_gradients(gradients, freq_dic[comp], axes_components[nr_comp], title=comp)

        if include_sum:
            comp_gradients[nr_comp] = gradients

    if include_sum:
        sum_gradients = np.sum(comp_gradients, axis=0)
        plot_gradients(sum_gradients, freq_dic[components[0]], axSum, title="sum")

    return fig, axes_components


def get_frequency_dic(feature_keys, get_select_dic=False):
    """
    get a dictionary with unique components of the feature keys (like Re, Im, magnitude ...)
    and the corresponding frequency array of that unique component
    @param feature_keys - list of strings in format <something>_<component>_<frequency>
    @return - dictionary with components as keys and frequency arrays as values (e.g. {"Re": [0.1, 1, 10], "Im": [0.1, 1, 10]})
    """
    # get frequencies and components from feature key strints
    # last part of string divided by "_" is frequency
    frequencies = [float(key.split("_")[-1]) for key in feature_keys]
    frequencies = np.array(frequencies)

    components = [str(key.split("_")[-2]) for key in feature_keys]
    components = np.array(components)
    components_unique = list(set(components))

    frequency_dic = {}
    select_dic = {}

    for comp in components_unique:
        select_comp = np.where(components == comp)[0]
        frequency_dic[comp] = frequencies[select_comp]
        select_dic[comp] = select_comp

    if get_select_dic:
        return frequency_dic, select_dic
    else:
        return frequency_dic
