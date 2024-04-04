# -*- coding: utf-8 -*-

import itertools
import multiprocessing
import os
import sys
import time
from io import StringIO
from math import pi

import numpy as np
import pandas as pd
import psutil
import scipy.stats as st
from scipy import interpolate
from scipy.signal import find_peaks
from IPython import display as IPdisplay

from impedance import preprocessing
from impedance.models.circuits import CustomCircuit
from impedance.models.circuits.fitting import rmse
from impedance.validation import linKK

import ipywidgets as widgets
from ipywidgets import fixed
from IPython.display import display

# # 'modules/eisplot' and 'modules/pyDRTtools/main/general_fun_v8' are local imports.
# sys.path.insert(0, r'modules/pyDRTtools/main/')
import modules.eisplot as eisplot
import modules.pyDRTtools.main.general_fun_v8 as general_fun

# The amount of CPU cores available for multiprocessing.
cores = psutil.cpu_count(logical=False)

# The sankey dictionary to store the amount of files and measurements.
sankey_dict = dict()


def get_df_csv(folder_path, index_columns=["cell_ID", "EIS_measurement_id"]):
    """
    Read a CSV file and set the index to the specified columns.

    Parameters:
        folder_path (str): The path to the CSV file.
        index_columns (list, optional): The columns to set as the index. Defaults to ["cell_ID", "EIS_measurement_id"].

    Returns:
        pd.DataFrame: The dataframe with the specified index columns.
    """
    df = pd.read_csv(folder_path)
    df.set_index(index_columns, inplace=True)
    return df


def search_filenames(folder_path, search_filters=[".",], file_end=".csv"):
    """
    Search for files in a folder with a specific extension.

    Parameters:
        folder_path (str): The folder path to search for files in.
        search_filters (list, optional): The file extensions to search for. Defaults to ["."].
        file_end (str, optional): The file extension to search for. Defaults to ".csv".

    Returns:
        list: A list of file names.
    """

    file_names = os.listdir(folder_path)
    csv_names = list(
        filter(lambda file_name: file_name.endswith(file_end), file_names))

    filtered_names = []
    for s_filter in search_filters:
        filtered_names += list(filter(lambda file_name: s_filter in file_name, csv_names))

    # Remove double entries
    filtered_names = list(set(filtered_names))

    sankey_dict.update({'all_files_in': len(csv_names),
                       'filtered_files_in': len(filtered_names)})

    return filtered_names


def get_df_folder(folder_path, file_names, index_columns=["cell_ID", "EIS_measurement_id", "EIS_Frequency"]):
    """
    Read multiple csv files from a folder and combine them into a single dataframe.

    Parameters:
        folder_path (str): The path to the folder containing the csv files.
        file_names (list): The list of file names in the folder.
        index_columns (list, optional): The columns to use as the index. Defaults to ["cell_ID", "EIS_measurement_id", "EIS_Frequency"].

    Returns:
        pd.DataFrame: The combined dataframe.
    """
    measurement_counter = 0
    df_list = []
    for csv_name in file_names:
        # Read the csv file
        filepath = os.path.join(folder_path, csv_name)
        df_new = pd.read_csv(filepath)
        # Add a new column "cell_ID" and fill it with the name of the csv
        df_new[index_columns[0]] = csv_name[:-4]

        df_list.append(df_new)

        measurement_counter += len(np.unique(df_new.EIS_measurement_id))

    df = pd.concat(df_list)
    df.EIS_Frequency = df.EIS_Frequency.astype('float64')
    df.set_index(index_columns, inplace=True)

    sankey_dict.update({'all_measurements_in': measurement_counter})
    return df


def add_cartesian(df, enable_degree=False):
    """
    Add the cartesian impedance values to the dataframe.

    Parameters:
        df (pd.DataFrame): The dataframe containing the impedance values.
        enable_degree (bool, optional): Whether or not to convert the phase to radians. Defaults to False.

    Returns:
        pd.DataFrame: The dataframe with the cartesian impedance values added.
    """
    abs_values = df["EIS_Z_abs"].to_numpy()
    phases = df["EIS_Z_phase"].to_numpy()
    if enable_degree:
        phases *= (pi/180)
    impedance_z = abs_values * np.exp(1j * phases)

    df.insert(df.columns.get_loc("EIS_Z_phase")+1,
              "EIS_Z_Re", np.real(impedance_z), True)
    df.insert(df.columns.get_loc("EIS_Z_Re")+1,
              "EIS_Z_Im", np.imag(impedance_z), True)

    return df


def plot_frequency_groups(fig, ax1, ax2, frequency_groups, df):
    """
    Plot the frequency groups and the violin plot for the frequency groups.

    Parameters:
        fig (matplotlib.figure.Figure): The figure to plot the frequency groups and violin plot on.
        ax1 (matplotlib.axes.Axes): The axis to plot the frequency groups on.
        ax2 (matplotlib.axes.Axes): The axis to plot the violin plot on.
        frequency_groups (list): The frequency groups to plot.
        df (pd.DataFrame): The dataframe containing all data.
    """

    x_values = [f_g_index * np.ones([len(f_group)])
                for f_g_index, f_group in enumerate(frequency_groups)]
    y_values = [np.log10(np.array(f_group)) for f_group in frequency_groups]

    # plot the data for each frequency group individually, one plot for each frequency group
    for ind in range(len(x_values)):
        ax1.scatter(x_values[ind], y_values[ind], marker='.', linewidth=0)
    ax1.set_xlabel("Group Index")
    ax1.set_ylabel("Frequency in Hz")
    ax1.grid(True)

    ax1.set_axisbelow(True)
    ax1.yaxis.set_major_formatter(
        eisplot.mpl.ticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
    ymin, ymax = ax1.get_ylim()
    tick_range = np.arange(ymin, ymax)
    ax1.yaxis.set_ticks(tick_range)
    ax1.yaxis.set_ticks([np.log10(x) for p in tick_range for x in np.linspace(
        10 ** p, 10 ** (p + 1), 10)], minor=True)
    ax1.set_ylabel("Frequency in Hz")
    ax1.grid(True)
    ax1.set_ylim([ymin, ymax])

    frequencies = df.index.droplevel("cell_ID").droplevel(
        "EIS_measurement_id").to_numpy()
    frequencies_log = np.log10(frequencies)

    x_jittered = st.t(df=6, scale=0.04).rvs(len(frequencies_log))
    ax2.scatter(x_jittered, frequencies_log, s=2,
                color=eisplot.rwth_colors.colors[('black', 50)], alpha=0.1)

    violins = ax2.violinplot(
        frequencies_log,
        positions=[0],
        bw_method="silverman",
        showextrema=False,
        points=500
    )
    for violin in violins['bodies']:
        violin.set_facecolor("none")
        violin.set_edgecolor(eisplot.rwth_colors.colors[('blue', 100)])
        violin.set_linewidth(1.5)
        violin.set_alpha(1)

    medianprops = dict(
        linewidth=1.5,
        color=eisplot.rwth_colors.colors[('black', 100)],
        solid_capstyle="butt"
    )
    boxprops = dict(
        linewidth=1.5,
        color=eisplot.rwth_colors.colors[('black', 100)]
    )

    ax2.set_axisbelow(True)
    ax2.boxplot(
        frequencies_log,
        positions=[0],
        showfliers=False,  # Do not show the outliers beyond the caps.
        showcaps=False,  # Do not show the caps
        medianprops=medianprops,
        whiskerprops=boxprops,
        boxprops=boxprops
    )

    ax2.yaxis.set_major_formatter(
        eisplot.mpl.ticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
    ymin, ymax = ax1.get_ylim()
    tick_range = np.arange(ymin, ymax)
    ax2.yaxis.set_ticks(tick_range)
    ax2.yaxis.set_ticks([np.log10(x) for p in tick_range for x in np.linspace(
        10 ** p, 10 ** (p + 1), 10)], minor=True)
    ax2.grid(True)
    ax2.set_ylim([ymin, ymax])

    ax2.get_xaxis().set_visible(False)
    ax2.get_xaxis().set_ticks([])

    ax2.scatter(0, np.mean(frequencies_log), s=10,
                color=eisplot.rwth_colors.colors[('red', 100)], zorder=3)

    ax2.set_title("Violin and Box plot of all Groups")

    eisplot.plt.subplots_adjust(bottom=0.14)


class FrequencySelectWidget:
    """
    Initialize the frequency selection widget.

    Parameters:
        axs (matplotlib.axes.Axes): The axes to plot on.
        fig (matplotlib.figure.Figure): The figure to plot on.
        df (pandas.DataFrame): The dataframe containing the data.
        frequency_groups_df (pandas.DataFrame): The frequency groups dataframe.
        frequency_groups (list): The frequency groups.

    Returns:
        frequency_select_widget: The frequency selection widget.
    """

    def __init__(self, axs, fig, df, frequency_groups_df, frequency_groups):
        """
        Initialize the frequency menu widget.

        Parameters:
            axs (matplotlib.axes.Axes): The axes to plot on.
            fig (matplotlib.figure.Figure): The figure to plot on.
            df (pandas.DataFrame): The dataframe containing the data.
            frequency_groups_df (pandas.DataFrame): The frequency groups dataframe.
            frequency_groups (list): The frequency groups.

        Returns:
            frequency_select_widget: The frequency menu widget.
        """
        # generate strings for the group selection based on the statistics extracted
        self.frequencies = []
        self.frequency_groups_df = frequency_groups_df
        self.freq_list_group_string = [("%d" % item)
                                       for item in frequency_groups_df.index]
        self.freq_list_occurrence_string = [
            ("%0.6f" % item) for item in frequency_groups_df.loc[:, 'occurrence']/np.sum(frequency_groups_df.loc[:, 'occurrence'])]
        self.freq_list_length_string = [
            ("%02d" % item) for item in frequency_groups_df.loc[:, 'length']]
        self.freq_list_min_string = [("%5.4f" % item)
                                     for item in frequency_groups_df.loc[:, 'min']]
        self.freq_list_max_string = [("%5.4f" % item)
                                     for item in frequency_groups_df.loc[:, 'max']]

        self.freq_list = np.empty(
            [1, len(self.freq_list_occurrence_string)], dtype="U")

        self.string_tmp = np.repeat('Group: ', len(self.freq_list))
        self.freq_list = np.core.defchararray.add(
            self.freq_list, self.string_tmp)
        self.freq_list = np.core.defchararray.add(
            self.freq_list, self.freq_list_group_string)

        self.string_tmp = np.repeat(', Occurrence: ', len(self.freq_list))
        self.freq_list = np.core.defchararray.add(
            self.freq_list, self.string_tmp)
        self.freq_list = np.core.defchararray.add(
            self.freq_list, self.freq_list_occurrence_string)

        self.string_tmp = np.repeat(', Length: ', len(self.freq_list))
        self.freq_list = np.core.defchararray.add(
            self.freq_list, self.string_tmp)
        self.freq_list = np.core.defchararray.add(
            self.freq_list, self.freq_list_length_string)

        self.string_tmp = np.repeat(', Min: ', len(self.freq_list))
        self.freq_list = np.core.defchararray.add(
            self.freq_list, self.string_tmp)
        self.freq_list = np.core.defchararray.add(
            self.freq_list, self.freq_list_min_string)

        self.string_tmp = np.repeat(', Max: ', len(self.freq_list))
        self.freq_list = np.core.defchararray.add(
            self.freq_list, self.string_tmp)
        self.freq_list = np.core.defchararray.add(
            self.freq_list, self.freq_list_max_string)

        self.freq_list_dic = dict(
            zip(self.freq_list.tolist()[0], frequency_groups_df.index.tolist()))

        self.dropdown = widgets.Dropdown(
            options=self.freq_list_dic,
            description='Frequency Group:',
            layout={'width': 'max-content'},
            style={'description_width': 'initial'}
        )

        self.slider_min = widgets.FloatLogSlider(
            min=np.max(
                [-2, np.floor(np.log10(np.min(pd.unique(df.index.get_level_values(2)))))]),
            max=np.ceil(
                np.log10(np.max(pd.unique(df.index.get_level_values(2))))),
            value=np.log10(
                np.max([0.01, np.min(pd.unique(df.index.get_level_values(2)))])),
            base=10,
            step=0.1,  # exponent step
            description='Min:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout_format='.6f',
            layout=widgets.Layout(width='20%'),
            style={'description_width': 'initial'},
        )

        self.slider_max = widgets.FloatLogSlider(
            value=np.min(
                [4500, np.max(pd.unique(df.index.get_level_values(2)))]),
            min=np.floor(
                np.log10(np.min(pd.unique(df.index.get_level_values(2))))),
            max=np.ceil(
                np.log10(np.max(pd.unique(df.index.get_level_values(2))))),
            base=10,
            step=0.01,  # exponent step
            description='Max:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout_format='.6f',
            layout=widgets.Layout(width='20%'),
            style={'description_width': 'initial'},
        )

        self.slider_f_per_decade = widgets.IntText(
            value=8,
            min=1,
            max=100,
            description='Frequencies per decade:',
            layout={'width': 'max-content'},
            style={'description_width': 'initial'},
            disabled=False
        )

        self.f_menu_sel = widgets.RadioButtons(
            options=['Frequency Group', 'Manuel'],
            value='Manuel',
            description='Select Method:',
            disabled=False,
            layout={'width': 'max-content'},
            style={'description_width': 'initial'}
        )

        self.coverage = widgets.FloatProgress(
            value=100.0,
            min=0,
            max=100.0,
            description='Remaining Frequencies: 100.00 %',
            bar_style='info',
            style={'bar_color': eisplot.rwth_colors.colors[(
                'blue', 100)], 'description_width': 'initial'},
            orientation='horizontal',
            layout=widgets.Layout(width='20%'),
        )

        frequency_groups_average_length = np.sum(
            self.frequency_groups_df.occurrence / np.sum(self.frequency_groups_df.occurrence) * self.frequency_groups_df.length)
        self.plot_hist_freq(df.index.get_level_values(
            2), df.index.get_level_values(2), axs)

        self.slider = widgets.HBox([self.slider_min, self.slider_max])
        self.f_menu_f_manual = widgets.VBox(
            [self.slider, self.slider_f_per_decade])
        self.f_menu_sel_me = widgets.VBox([self.f_menu_sel, self.coverage])
        self.f_menu_method = widgets.VBox([self.f_menu_sel_me, self.dropdown])
        self.ui = widgets.VBox([self.f_menu_method, self.f_menu_f_manual])

        # now combine everything to an interactive output
        self.out = widgets.interactive_output(self.update_f,
                                              {'axs': fixed(axs), 'fig': fixed(fig), 'df': fixed(df),
                                               'frequency_groups': fixed(frequency_groups), 'f_menu_sel_p': self.f_menu_sel,
                                               'slider_min_p': self.slider_min, 'slider_max_p': self.slider_max,
                                               'dropdown_p': self.dropdown,
                                               'slider_f_per_decade_p': self.slider_f_per_decade})

        display(self.ui, self.out)

    def plot_hist_freq(self, all_frequencies, selected_frequencies, ax, label='', alpha=1.0):
        """
        Plot the histogram of the frequencies of the data.

        Parameters:
            all_frequencies (array-like): The frequencies of the data.
            selected_frequencies (array-like): The frequencies to plot.
            ax (matplotlib.axes.Axes): The axis to plot on.
            label (str): The label for the plot.
            alpha (float): The alpha value for the plot.
        """
        hist, bin_edges = np.histogram(all_frequencies, bins=np.append(
            np.sort(np.unique(selected_frequencies)), np.max(selected_frequencies)))
        hist = hist / np.sum(hist)
        center = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.bar(center, hist, width=np.diff(bin_edges)
               * 0.8, label=label, alpha=alpha)

        ax.grid(True)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel("Frequency in Hz")
        ax.set_ylabel("Normalized probability")
        eisplot.plt.subplots_adjust(bottom=0.14)

    def update_coverage(self, percentage):
        """
        Update the coverage widget with the current percentage of frequencies remaining.

        Parameters:
            percentage (float): The percentage of frequencies remaining.
        """
        self.coverage.description = 'Remaining Frequencies: ' + \
            ("%02.2f" % percentage) + ' %'
        self.coverage.value = percentage

    def cal_coverage(self, f_min, f_max, frequency_groups_df):
        """
        Calculate the coverage of a frequency group.

        Parameters:
            f_min (float): The minimum frequency of the frequency group.
            f_max (float): The maximum frequency of the frequency group.
            frequency_groups_df (pandas.DataFrame): The frequency groups dataframe.

        Returns:
            float: The coverage of the frequency group.
        """
        total_count = np.sum(frequency_groups_df.loc[:, 'occurrence'])
        covered_count = np.sum(((frequency_groups_df['min'] <= f_min) & (
            frequency_groups_df['max'] >= f_max)) * frequency_groups_df.loc[:, 'occurrence'])
        return covered_count / total_count * 100

    def update_f(self, axs, fig, df, frequency_groups, f_menu_sel_p='Frequency Group', slider_min_p=0.1, slider_max_p=1000, dropdown_p=0, slider_f_per_decade_p=6):
        """
        Update the frequency plot.

        Parameters:
            axs (matplotlib.axes.Axes): The frequency plot.
            fig (matplotlib.figure.Figure): The frequency plot figure.
            df (pandas.DataFrame): The dataframe containing the data.
            frequency_groups (list): The frequency groups.
            f_menu_sel_p (str): The selected Frequency Group.
            slider_min_p (float): The minimum frequency.
            slider_max_p (float): The maximum frequency.
            dropdown_p (int): The selected Frequency Group.
            slider_f_per_decade_p (int): The number of frequencies.

        Returns:
            None
        """

        frequency_groups_average_length = np.sum(self.frequency_groups_df.occurrence / np.sum(
            self.frequency_groups_df.occurrence) * self.frequency_groups_df.length)
        if f_menu_sel_p == 'Frequency Group':
            self.dropdown.disabled = False
            self.slider_min.disabled = True
            self.slider_max.disabled = True
            self.slider_f_per_decade.disabled = True

            self.frequencies = np.array(frequency_groups[self.dropdown.value])
            self.update_coverage(self.cal_coverage(np.min(frequency_groups[self.dropdown.value]), np.max(
                frequency_groups[self.dropdown.value]), self.frequency_groups_df))

        else:
            self.dropdown.disabled = True
            self.slider_min.disabled = False
            self.slider_max.disabled = False
            self.slider_f_per_decade.disabled = False

            self.frequencies = np.logspace(np.log10(self.slider_min.value), np.log10(self.slider_max.value), round(
                np.ceil((np.log10(self.slider_max.value) - np.log10(self.slider_min.value))) * self.slider_f_per_decade.value))
            self.update_coverage(self.cal_coverage(
                slider_min_p, slider_max_p, self.frequency_groups_df))

        axs.cla()
        self.plot_hist_freq(df.index.get_level_values(2), df.index.get_level_values(2),
                            axs, label='Frequencies in the Data', alpha=1.0)
        self.plot_hist_freq(self.frequencies, self.frequencies,
                            axs, label='Selected Frequencies', alpha=0.5)
        axs.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=2)
        fig.canvas.draw_idle()


class LinkkSelectWidget:
    """
    Initialize the LinkkSelectWidget class.

    Parameters:
        ax1 (matplotlib.axes.Axes): The axis to plot on.
        ax2 (matplotlib.axes.Axes): The axis to plot on.
        fig (matplotlib.figure.Figure): The figure to plot on.
        df (pandas.DataFrame): The dataframe to plot on.
    """

    def __init__(self, ax1, ax2, fig, df):
        """
        Initialize the linKK plot.

        Parameters:
            ax1 (matplotlib.axes.Axes): The axis to plot on.
            ax2 (matplotlib.axes.Axes): The axis to plot on.
            fig (matplotlib.figure.Figure): The figure to plot on.
            df (pandas.DataFrame): The dataframe to plot on.
        """
        self.linKK_limit = 0

        self.slider_linKK = widgets.FloatSlider(
            value=np.max(pd.unique(df["linKK"])),
            min=0,
            max=np.max(pd.unique(df["linKK"])),
            step=0.00001,
            description='linKK Limit:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.5f',
            layout=widgets.Layout(width='40%'),
            style={'description_width': 'initial'}
        )

        self.coverage_linKK = widgets.FloatProgress(
            value=100.0,
            min=0,
            max=100.0,
            description='Remaining Measurements: 100.00 %',
            bar_style='info',
            style={'bar_color': eisplot.rwth_colors.colors[(
                'blue', 100)], 'description_width': 'initial'},
            orientation='horizontal',
            layout=widgets.Layout(width='20%'),
        )

        self.plot_linKK(np.median(df.groupby(level=[0, 1])[
                        'linKK'].mean().to_numpy()), df, ax1, alpha=1.0)

        self.ui = widgets.VBox([self.coverage_linKK, self.slider_linKK])
        self.out = widgets.interactive_output(self.update_linKK, {'ax1': fixed(ax1), 'ax2': fixed(
            ax2), 'fig': fixed(fig), 'df': fixed(df), 'slider_linKK_p': self.slider_linKK})
        display(self.ui, self.out)

    def update_coverage_linKK(self, percentage):
        """
        Update the coverage linKK progress bar with the new percentage.

        Parameters:
            percentage (float): The new percentage of coverage linKK.
        """
        self.coverage_linKK.description = 'Remaining Measurements: ' + \
            ("%02.2f" % percentage) + ' %'
        self.coverage_linKK.value = percentage

    def cal_coverage_linKK(self, linKK_limit, df):
        """
        Calculate the coverage of the linKK values for the given linKK limit and dataframe.

        Parameters:
            linKK_limit (float): The linKK limit to compare against.
            df (pandas.DataFrame): The dataframe to compare against.

        Returns:
            float: The coverage of the linKK values for the given linKK limit and dataframe.
        """
        total_count = np.sum(df.groupby(level=[0, 1])[
                             'linKK'].mean().to_numpy() >= 0)
        covered_count = np.sum(df.groupby(level=[0, 1])[
                               'linKK'].mean().to_numpy() <= linKK_limit)
        return covered_count / total_count * 100

    def plot_linKK(self, linKK_limit, df, ax, alpha=1.0):
        """
        Plot the linKK of the measurements.

        Parameters:
            linKK_limit (float): The limit of the linKK plot.
            df (pandas.DataFrame): The dataframe of the measurements.
            ax (matplotlib.axes.Axes): The axis to plot on.
            alpha (float): The alpha of the plot.
        """
        ax.scatter(range(len(df.groupby(level=[0, 1])['linKK'].mean().to_numpy())),
                   np.sort(df.groupby(level=[0, 1])[
                           'linKK'].mean().to_numpy()),
                   label="linKK of Measurements",
                   alpha=alpha,
                   marker='.',
                   linewidth=0,
                   color=eisplot.rwth_colors.colors[('blue', 100)])
        ax.set_xlabel("EIS Measurement")
        ax.set_xticks([])
        ax.set_ylabel(
            "Quantile/Max Absolute \nresidual of the measurement in Ohm")
        ax.grid(True)
        linKK_max = np.ceil(np.max(df.groupby(level=[0, 1])[
                            'linKK'].mean().to_numpy()) * 100) / 100
        ax.plot(range(len(df.groupby(level=[0, 1])['linKK'].mean().to_numpy())),
                np.ones(
                    [len(df.groupby(level=[0, 1])['linKK'].mean().to_numpy()), 1]) * linKK_limit,
                label="Limit",
                alpha=alpha,
                color=eisplot.rwth_colors.colors[('black', 100)])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5)

    def plot_linKK_stats(self, linKK_limit, df, ax, alpha=1.0):
        """
        Plot the linKK statistics of the measurements.

        Parameters:
            linKK_limit (float): The limit of the linKK plot.
            df (pandas.DataFrame): The dataframe of the measurements.
            ax (matplotlib.axes.Axes): The axis to plot on.
            alpha (float): The alpha of the plot.
        """
        x_data = np.arange(len(df.groupby(level=[0, 1])[
                           'linKK'].mean().to_numpy()))
        y_data = df.groupby(level=[0, 1])['linKK'].mean().to_numpy()

        x_jittered = st.t(df=6, scale=0.04).rvs(len(y_data))
        ax.scatter(x_jittered, y_data, s=2,
                   color=eisplot.rwth_colors.colors[('black', 50)], alpha=0.1)

        violins = ax.violinplot(
            y_data,
            positions=[0],
            bw_method="silverman",
            showextrema=False,
            points=5000
        )

        for violin in violins['bodies']:
            violin.set_facecolor("none")
            violin.set_edgecolor(eisplot.rwth_colors.colors[('lime', 100)])
            violin.set_linewidth(1.5)
            violin.set_alpha(0.5)

        medianprops = dict(
            linewidth=1.5,
            color=eisplot.rwth_colors.colors[('petrol', 100)],
            solid_capstyle="butt"
        )
        boxprops = dict(
            linewidth=1.5,
            color=eisplot.rwth_colors.colors[('petrol', 100)]
        )

        ax.set_axisbelow(True)
        ax.boxplot(
            y_data,
            positions=[0],
            showfliers=False,  # Do not show the outliers beyond the caps.
            showcaps=False,  # Do not show the caps
            medianprops=medianprops,
            whiskerprops=boxprops,
            boxprops=boxprops
        )

        ax.grid(True)

        ax.get_xaxis().set_visible(False)
        ax.get_xaxis().set_ticks([])

        ax.scatter(0, np.mean(y_data), s=10,
                   color=eisplot.rwth_colors.colors[('red', 100)], zorder=3)

        ax.plot(ax.get_xlim(), np.ones([len(ax.get_xlim()), 1]) * linKK_limit, label="Limit",
                color=eisplot.rwth_colors.colors[('black', 100)], alpha=alpha)

        ax.set_title("Violin and Box plot \n of all EIS Measurements")

        eisplot.plt.subplots_adjust(bottom=0.14)

    def update_linKK(self, ax1, ax2, fig, df, slider_linKK_p=0.1):
        """
        Update the linKK plot with the new value of the slider. Also update the coverage plot.

        Parameters:
            ax1 (matplotlib.axes.Axes): The axis of the linKK plot.
            ax2 (matplotlib.axes.Axes): The axis of the linKK plot.
            fig (matplotlib.figure.Figure): The figure of the linKK plot.
            df (pandas.DataFrame): The dataframe of the linKK plot.
            slider_linKK_p (float): The value of the slider.
        """
        self.slider_linKK.disabled = False
        self.coverage_linKK.disabled = False

        self.slider_linKK.max = np.ceil(
            np.max(df.groupby(level=[0, 1])['linKK'].mean().to_numpy()) * 100) / 100

        self.linKK_limit = self.slider_linKK.value

        ax1.cla()
        self.update_coverage_linKK(
            self.cal_coverage_linKK(self.slider_linKK.value, df))
        self.plot_linKK(self.slider_linKK.value, df, ax1, alpha=1.0)

        ax2.cla()
        self.plot_linKK_stats(self.slider_linKK.value, df, ax2, alpha=1.0)

        fig.canvas.draw_idle()


def indexes_of_level(df, level):
    """
    Given a dataframe and a level, return a list of the indexes of that level.

    Parameters:
        df (pandas.DataFrame): The dataframe we are working with.
        level (int): The level we are working with.

    Returns:
        list: The list of indexes of that level.
    """
    return list(set(df.index.get_level_values(level)))


def filter_shapes(df, frequencies):
    """
    Filter out shapes in the dataframe that are not the same length as the list of frequencies.

    Parameters:
        df (pandas.DataFrame): The dataframe containing the shapes.
        frequencies (list): The list of frequencies.

    Returns:
        pandas.DataFrame: The filtered dataframe.
    """
    freq_shape = len(frequencies)

    # Get all measurement IDs
    index_IDs = df.index.get_level_values(1).to_numpy()

    # Split the measurement IDs into arrays
    ID_changes = index_IDs[1:] - index_IDs[:-1]
    split_IDs = np.where(ID_changes == 1)[0] + 1
    splitted_IDs = np.split(index_IDs, split_IDs)

    # Use the split to divide the indexes
    splitted_indexes = np.split(df.index.to_numpy(), split_IDs)

    # Get an array with all measurement shapes
    shape_list = [len(array) for array in splitted_IDs]
    shape_arr = np.array(shape_list)

    # Find measurement shapes that fit the freq_shape
    drop_measurements = np.where(shape_arr != freq_shape)[0]

    # Get the indexes of the right shaped measurements
    drop_list = [splitted_indexes[drop_index]
                 for drop_index in drop_measurements]
    drop_list = list(np.concatenate(drop_list))
    drop_indexes = pd.MultiIndex.from_tuples(drop_list)

    return df.drop(drop_indexes)


def filter_frequencies(df, frequencies, relative_tolerance=1e-03):
    """
    Filter the dataframe based on the frequency of the shapes.

    Parameters:
        df (pandas.DataFrame): The dataframe containing the data.
        frequencies (list): The frequencies of the shapes.
        relative_tolerance (float): The relative tolerance of the frequencies.

    Returns:
        pandas.DataFrame: The filtered dataframe.
    """
    # get deep copy of df
    print("initial datapoints:", len(df))
    df = filter_shapes(df, frequencies)
    print("datapoints after shape filter:", len(df))

    # go through every measurement and check if measurement frequencies align with given frequencies
    # if they do not align, save the locations of measurement and delete it later
    drop_locs = []
    for cell_ID in indexes_of_level(df, 0):  # go through cells
        df_cell = df.loc[cell_ID]
        # go through sweeps in cell
        for measure_ID in indexes_of_level(df_cell, 0):
            # get the measured frequencies
            measure_freq = df_cell.loc[measure_ID].index.to_numpy()

            # check if number of sweep frequencies align
            if len(measure_freq) == len(frequencies):
                # save the location of the measurement to drop if frequencies do not align
                if not np.allclose(frequencies, measure_freq, rtol=relative_tolerance):
                    drop_loc = df.index.get_locs((cell_ID, measure_ID))
                    drop_locs.append(drop_loc)
            else:
                # save the location of the measurement to drop
                drop_loc = df.index.get_locs((cell_ID, measure_ID))
                drop_locs.append(drop_loc)

    # make drop locs single np array
    drop_locs = np.concatenate(drop_locs)

    # get the drop indexes
    drop_indexes = df.index[drop_locs]

    # drop the indexes
    df_filtered = df.drop(drop_indexes)
    print("datapoints frequency filter:", len(df))
    return df_filtered


def restruct_df(df, frequencies, eis_column_names=["EIS_Z_abs", "EIS_Z_phase"], relative_tolerance=1e-03):
    """
    Restructures the given dataframe to be in the format of the EIS data.

    Parameters:
        df (pandas.DataFrame): The dataframe to be restructured.
        frequencies (list): The frequencies of the dataframe.
        eis_column_names (list): The column names of the EIS dataframe.
        relative_tolerance (float): The relative tolerance of the dataframe.

    Returns:
        pandas.DataFrame: The restructured dataframe.
    """

    all_keys = list(df.keys())

    # all keys that can be averaged over a sweep (not eis keys)
    average_keys = list(
        filter(lambda key: key not in eis_column_names, all_keys))

    # set column names for new DataFrame
    eis_columns = average_keys + get_eis_columns(eis_column_names, frequencies)

    # select individual measurements as index for new DataFrame
    measure_ids = df.index.droplevel(2)
    measure_ids = measure_ids[np.invert(measure_ids.duplicated(keep="first"))]

    # create empty new Dataframe
    df_r = pd.DataFrame(columns=eis_columns, index=measure_ids)

    # make empty numpy array for averaging with right dimensions
    nr_params = len(average_keys)  # nr of parameters to average
    nr_measurements = len(measure_ids)  # nr of single frequency measurements
    nr_freq = len(frequencies)  # nr of frequencies
    average_params = np.zeros((nr_params, nr_measurements, nr_freq))

    # go through every frequency to fill the corresponding column
    # in the new DataFrame with impedances

    index_freqs = df.index.get_level_values(2).to_numpy()
    for freq_count, freq in enumerate(frequencies):
        # select part of df where eis frequency is close to freq
        df_freq = df.loc[np.isclose(
            index_freqs, freq, rtol=relative_tolerance)]

        # check if dimensions match up
        e_str = "Can't select the frequencies. Try different relative_tolerance."
        assert len(df_freq) == nr_measurements, e_str

        # save parameters for averaging (later) into 3d array
        for av_count, av_key in enumerate(average_keys[1:]):
            if pd.to_numeric(df_freq[av_key], errors='coerce').notnull().all():
                average_params[av_count, :,
                               freq_count] = df_freq[av_key].to_numpy()
            else:
                average_params[av_count, :,
                               freq_count] = df_freq[av_key].values[0]

        # go through eis columns (except frequency) and copy values
        for eis_key in eis_column_names:
            col_key = get_column_key(eis_key, freq)
            df_r[col_key] = df_freq[eis_key].to_numpy()

    # fill non-eis columns with the mean values over the eis sweeps
    param_means = average_params.mean(axis=-1)
    for av_count, av_key in enumerate(average_keys[1:]):
        df_r[av_key] = param_means[av_count]

    # put the beginning times of the sweep into new column
    begin_times = df.loc[np.isclose(index_freqs, frequencies[0])]["Time"]
    df_r["Time"] = begin_times.to_numpy(dtype=np.datetime64)

    return df_r


def get_eis_columns(eis_column_names, frequencies):
    """
    Given a list of EIS column names and a list of frequencies, return a list of column names.

    Parameters:
        eis_column_names (list): The EIS column names.
        frequencies (list): The frequencies.

    Returns:
        list: The column names.
    """
    eis_columns = []
    for freq in frequencies:
        for eis_key in eis_column_names:
            column_name = get_column_key(eis_key, freq)
            eis_columns.append(column_name)

    return eis_columns


def get_column_key(eis_key, freq):
    """
    Given an EIS key and a frequency, return the column key for that EIS key and frequency.

    Parameters:
        eis_key (str): The EIS key.
        freq (float): The frequency.

    Returns:
        str: The column key for that EIS key and frequency.
    """
    return str(eis_key + "_" + str(freq)[:9])


def get_key_lookup_df(eis_column_names, frequencies):
    """
    Given a list of EIS column names and a list of frequencies, create a dataframe with the
    frequency as the index and the EIS column names as the columns. The values in the dataframe
    are the keys for the EIS column names.

    Parameters:
        eis_column_names (list): The EIS column names.
        frequencies (list): The frequencies.

    Returns:
        pandas.DataFrame: The dataframe with the frequency as the index and the EIS column names as the columns.
    """
    new_data = np.empty(
        [len(frequencies), len(eis_column_names)], dtype="<U1000")

    for freq_ind, freq in enumerate(frequencies):
        for eis_col_ind, eis_column_name in enumerate(eis_column_names):
            new_data[freq_ind, eis_col_ind] = get_column_key(
                eis_column_name, freq)

    key_lookup_df = pd.DataFrame(
        new_data, index=frequencies, columns=eis_column_names)
    key_lookup_df.index.name = 'frequency'

    return key_lookup_df


def add_ECMfit_parallel_func(df, frequencies, abs_keys, phase_keys, c_circuit, c, global_fit=False, maxfev=4000):
    """
    Fit the circuit to the data in parallel for each cell_ID and EIS_measurement_id.

    Parameters:
        df (pandas.DataFrame): The dataframe containing the data to be fit.
        frequencies (list): The frequencies to be fit.
        abs_keys (list): The keys for the absolute values.
        phase_keys (list): The keys for the phases.
        c_circuit (CustomCircuit): The circuit to be fit.
        c (list): The keys for the circuit parameters.
        global_fit (bool): Whether to perform a global fit. Default is False.
        maxfev (int): The maximum number of function evaluations. Default is 4000.

    Returns:
        pandas.DataFrame: The dataframe with the ECM fit added.
    """
    for cell_ID_loop in df.index.unique("cell_ID"):
        for EIS_measurement_id_loop in df.loc[cell_ID_loop].index.unique("EIS_measurement_id"):
            if len(frequencies) > len(c)+2:  # bad error handling
                abs_values = df.loc[cell_ID_loop].loc[EIS_measurement_id_loop][abs_keys].to_numpy(
                    dtype='float64')  # important to use dtype='float64'
                phases = df.loc[cell_ID_loop].loc[EIS_measurement_id_loop][phase_keys].to_numpy(
                    dtype='float64')  # important to use dtype='float64'
                impedance_z = abs_values * np.exp(1j * phases)
                impedance_z = impedance_z.astype(complex)

                # run fit
                mask = (df.index.get_level_values(0) == cell_ID_loop) & (
                    df.index.get_level_values(1) == EIS_measurement_id_loop)

                try:
                    if global_fit == False:
                        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html#scipy.optimize.leastsq
                        c_circuit.fit(frequencies, impedance_z,
                                      global_opt=global_fit, maxfev=maxfev)
                    else:
                        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html
                        c_circuit.fit(frequencies, impedance_z,
                                      global_opt=global_fit, niter=maxfev)

                    impedance_z_fit = c_circuit.predict(frequencies)
                    impedance_z_fit_rmse = rmse(impedance_z, impedance_z_fit)

                    for ind, value in enumerate(c):
                        df.loc[mask, value] = c_circuit.parameters_[ind]

                    df.loc[mask, 'ECM_Fit_RMSE'] = impedance_z_fit_rmse
                except:
                    for ind, value in enumerate(c):
                        df.loc[mask, value] = np.nan
                    df.loc[mask, 'ECM_Fit_RMSE'] = np.nan
    
    return df


def add_ECMfit(df, key_lookup_df, circuit='L0-R0-p(CPE1,R1)-p(CPE2,R2)-p(CPE3,R3)', initial_guess=[1.0e-08, 1.0e-02, 1.0e-01, 0.5e+00, 1.0e-03, 1.0e+01, 0.5e+00, 1.0e-01, 1.0e+03, 0.5e+00, 1.0e+01], global_fit=False, maxfev=4000):
    """
    Add the ECM fit to the dataframe.

    Parameters:
        df (pandas.DataFrame): The dataframe containing the data.
        key_lookup_df (pandas.DataFrame): The dataframe containing the key lookup.
        circuit (str): The circuit to use. Default is 'L0-R0-p(CPE1,R1)-p(CPE2,R2)-p(CPE3,R3)'.
        initial_guess (list): The initial guess for the fit. Default is [1.0e-08, 1.0e-02, 1.0e-01, 0.5e+00, 1.0e-03, 1.0e+01, 0.5e+00, 1.0e-01, 1.0e+03, 0.5e+00, 1.0e+01].
        global_fit (bool): Whether to do a global fit. Default is False.
        maxfev (int): The maximum number of function evaluations. Default is 4000.

    Returns:
        pandas.DataFrame: The dataframe with the ECM fit added.
    """
    c_circuit = CustomCircuit(circuit, initial_guess=initial_guess)

    c = c_circuit.get_param_names()[0]
    c = ["ECM_"+name for name in c]
    df = df.assign(**dict.fromkeys(c, 42.0))
    df = df.assign(ECM_Fit_RMSE=42.0)

    frequencies = key_lookup_df["frequency"].to_numpy()

    abs_keys = key_lookup_df["EIS_Z_abs"].to_list()
    phase_keys = key_lookup_df["EIS_Z_phase"].to_list()

    # split the dataframe into chunks for parallel processing, and then process each chunk individually
    pool = multiprocessing.Pool(cores)

    df_list = list(np.array_split(df, cores))
    frequencies_list = list(itertools.repeat(frequencies, cores))
    abs_keys_list = list(itertools.repeat(abs_keys, cores))
    phase_keys_list = list(itertools.repeat(phase_keys, cores))
    c_circuit_list = list(itertools.repeat(c_circuit, cores))
    c_list = list(itertools.repeat(c, cores))
    global_fit_list = list(itertools.repeat(global_fit, cores))
    maxfev_list = list(itertools.repeat(maxfev, cores))

    results = pool.starmap(add_ECMfit_parallel_func, zip(df_list, frequencies_list,
                           abs_keys_list, phase_keys_list, c_circuit_list, c_list, global_fit_list, maxfev_list))
    df = pd.concat(results)
    pool.close()

    return df


def add_linKK_parallel_func(df, linKK_cutoff_mu=0.85, linKK_max_M=100, residuals_quantile=1):
    """
    Given a dataframe, add linKK values to the dataframe.

    Parameters:
        df (pandas.DataFrame): The dataframe we are working with.
        linKK_cutoff_mu (float): The cutoff value for the linKK fit. Default is 0.85.
        linKK_max_M (int): The maximum number of harmonics to fit. Default is 100.
        residuals_quantile (float): The quantile of the residuals to use for the linKK cutoff. Default is 1.

    Returns:
        pandas.DataFrame: The dataframe with linKK values added.
    """
    for cell_ID_loop in df.index.unique("cell_ID"):
        for EIS_measurement_id_loop in df.loc[cell_ID_loop].index.unique("EIS_measurement_id"):
            frequency = df.loc[cell_ID_loop].loc[EIS_measurement_id_loop].index.to_numpy(
            )
            abs_values = df.loc[cell_ID_loop].loc[EIS_measurement_id_loop].loc[:, 'EIS_Z_abs'].to_numpy(
                dtype='float64')
            phases = df.loc[cell_ID_loop].loc[EIS_measurement_id_loop].loc[:,
                                                                           'EIS_Z_phase'].to_numpy(dtype='float64')
            impedance_z = abs_values * np.exp(1j * phases)

            # filter for imag < 0
            frequency = frequency[np.imag(impedance_z) < 0]
            impedance_z = impedance_z[np.imag(impedance_z) < 0]

            if len(frequency) > 1:
                M, mu, Z_linKK, res_real, res_imag = linKK(
                    frequency, impedance_z, c=linKK_cutoff_mu, max_M=linKK_max_M, fit_type='complex', add_cap=True)
                residuals = res_real + 1j * res_imag
                residuals = np.abs(residuals)
                linKK_error = np.quantile(residuals, residuals_quantile)

                mask = (df.index.get_level_values(0) == cell_ID_loop) & (
                    df.index.get_level_values(1) == EIS_measurement_id_loop)
                df.loc[mask, 'linKK'] = linKK_error

    return df


def add_linKK(df, linKK_cutoff_mu=0.85, linKK_max_M=100, residuals_quantile=1):
    """
    Add the linKK column to the dataframe.

    Parameters:
        df (pandas.DataFrame): The dataframe to add the linKK column to.
        linKK_cutoff_mu (float): The cutoff value for the linKK calculation. Default is 0.85.
        linKK_max_M (int): The maximum number of measurements to use in the linKK calculation. Default is 100.
        residuals_quantile (float): The quantile of the residuals to use in the linKK calculation. Default is 1.

    Returns:
        pandas.DataFrame: The dataframe with the linKK column added.
    """
    df = df.assign(linKK=42.0)
    pool = multiprocessing.Pool(cores)

    # split the dataframe into chunks for parallel processing, one chunk per core
    cell_measurement_list = df.index.droplevel("EIS_Frequency").unique()
    cell_names_list = np.array_split(cell_measurement_list, cores)
    df_new_index = df.reset_index(level=['EIS_Frequency'])
    df_list = [df_new_index.loc[cell_names_list[i]].set_index(
        ['EIS_Frequency'], append=True) for i in range(len(cell_names_list))]
    linKK_cutoff_mu_list = [linKK_cutoff_mu] * cores
    linKK_max_M_list = [linKK_max_M] * cores
    residuals_quantile_list = [residuals_quantile] * cores

    results = pool.starmap(add_linKK_parallel_func, zip(
        df_list, linKK_cutoff_mu_list, linKK_max_M_list, residuals_quantile_list))
    df = pd.concat(results)
    pool.close()

    return df


def get_frequency_groups_parallel_func(df):
    """
    Given a dataframe, group the data by the frequency and return a dictionary of the groups.

    Parameters:
        df (pandas.DataFrame): The dataframe containing the data.

    Returns:
        tuple: A tuple containing the frequency groups dataframe and the frequency groups list.
    """
    df_fs = df.index
    df_fs = pd.DataFrame(df_fs.tolist(), columns=[
                         'cell_ID', 'EIS_measurement_id', 'EIS_Frequency'])
    all_cell_ids = pd.unique(df_fs.cell_ID)

    frequency_groups = list()
    frequency_groups_occurrence = []

    for cell_index, cell_name in enumerate(all_cell_ids):
        all_measurements = pd.unique(
            df_fs.loc[df_fs.cell_ID == cell_name].EIS_measurement_id)
        for m_id_index, m_id in enumerate(all_measurements):
            f_list_of_this_m = df_fs.loc[df_fs.cell_ID ==
                                         cell_name].loc[df_fs.EIS_measurement_id == m_id].EIS_Frequency.to_list()
            if f_list_of_this_m in frequency_groups:
                freq_group_ind = frequency_groups.index(f_list_of_this_m)
                frequency_groups_occurrence[freq_group_ind] = frequency_groups_occurrence[freq_group_ind] + 1
            else:
                frequency_groups.append(f_list_of_this_m)
                frequency_groups_occurrence.append(1)

    frequency_groups_length = np.array([len(x) for x in frequency_groups])
    # don't normalise here, by this you would loose informations /np.sum(frequency_groups_occurrence)
    frequency_groups_occurrence = np.array(frequency_groups_occurrence)
    frequency_groups_min = np.array([np.min(x) for x in frequency_groups])
    frequency_groups_max = np.array([np.max(x) for x in frequency_groups])

    frequency_groups_df = pd.DataFrame(np.transpose([frequency_groups_occurrence, frequency_groups_length,
                                       frequency_groups_min, frequency_groups_max]), columns=['occurrence', 'length', 'min', 'max'])
    frequency_groups_df = frequency_groups_df.sort_values(
        by=['occurrence', 'length', 'min', 'max'], ascending=False)

    return frequency_groups_df, frequency_groups


def get_frequency_groups(df):
    """
    Given a dataframe, group the data by frequency and return a dataframe with the frequency groups.

    Parameters:
        df (pandas.DataFrame): The dataframe containing the data.

    Returns:
        tuple: A tuple containing the frequency groups dataframe and the frequency groups list.
    """
    cell_measurement_list = df.index.droplevel("EIS_Frequency").unique()

    # split the dataframe into chunks of the same size as the number of cores available on the machine, and then run the function on each chunk individually
    cell_names_list = np.array_split(cell_measurement_list, cores)
    df_copy = df.copy()
    df_new_index = df_copy.reset_index(level=['EIS_Frequency'])
    df_list = [df_new_index.loc[cell_names_list[i]].set_index(
        ['EIS_Frequency'], append=True) for i in range(len(cell_names_list))]

    pool = multiprocessing.Pool(cores)
    results = pool.map(get_frequency_groups_parallel_func, df_list)

    pool.close()
    # concatenate all the results together and reset the index to start at 0 again
    frequency_groups_df_split = pd.concat([i[0] for i in results])
    frequency_groups_df_split = frequency_groups_df_split.reset_index(
        drop=True)

    frequency_groups_split = []
    for res in results:
        for freq_group in res[1]:
            frequency_groups_split.append(freq_group)

    frequency_groups = []
    frequency_groups_occurrence = []

    # if the frequency group is in the list of frequency groups increase the occurrence, otherwise add it
    for freq_group_ind_split, freq_group in enumerate(frequency_groups_split):
        if freq_group in frequency_groups:
            freq_group_ind_new = frequency_groups.index(freq_group)
            frequency_groups_occurrence[freq_group_ind_new] += frequency_groups_df_split.iloc[freq_group_ind_split]["occurrence"]
        else:
            frequency_groups.append(freq_group)
            frequency_groups_occurrence.append(
                frequency_groups_df_split.iloc[freq_group_ind_split]["occurrence"])

    frequency_groups_length = np.array([len(x) for x in frequency_groups])
    # don't normalize here, by this you would lose information /np.sum(frequency_groups_occurrence)
    frequency_groups_occurrence = np.array(frequency_groups_occurrence)
    frequency_groups_min = np.array([np.min(x) for x in frequency_groups])
    frequency_groups_max = np.array([np.max(x) for x in frequency_groups])

    frequency_groups_df = pd.DataFrame(np.transpose([frequency_groups_occurrence, frequency_groups_length,
                                       frequency_groups_min, frequency_groups_max]), columns=['occurrence', 'length', 'min', 'max'])
    frequency_groups_df = frequency_groups_df.sort_values(
        by=['occurrence', 'length', 'min', 'max'], ascending=False)

    return frequency_groups_df, frequency_groups


def filer_by_frequencies_groups_parallel_func(df, frequencies, minimal_amount_of_frequencies):
    """
    Filter the dataframe by the frequencies groups. This function is used in parallel.

    Parameters:
        df (pandas.DataFrame): The dataframe to filter.
        frequencies (list): The frequencies to filter by.
        minimal_amount_of_frequencies (int): The minimal amount of frequencies to keep.

    Returns:
        pandas.DataFrame: The filtered dataframe.
    """

    for cell_ID_loop in df.index.unique("cell_ID"):
        for EIS_measurement_id_loop in df.loc[cell_ID_loop].index.unique("EIS_measurement_id"):
            if (np.max(df.loc[cell_ID_loop].loc[EIS_measurement_id_loop].index) < np.max(frequencies)) or (np.min(df.loc[cell_ID_loop].loc[EIS_measurement_id_loop].index) > np.min(frequencies)):
                print("Dropped Cell: "+cell_ID_loop +
                      ", Measurement: "+str(EIS_measurement_id_loop))
                sys.stdout.flush()
                df = df.reset_index(level="EIS_Frequency").drop(index=(
                    cell_ID_loop, EIS_measurement_id_loop)).set_index(['EIS_Frequency'], append=True)
            elif len(df.loc[cell_ID_loop].loc[EIS_measurement_id_loop].index) < minimal_amount_of_frequencies:
                print("Dropped Cell: "+cell_ID_loop +
                      ", Measurement: "+str(EIS_measurement_id_loop))
                sys.stdout.flush()
                df = df.reset_index(level="EIS_Frequency").drop(index=(
                    cell_ID_loop, EIS_measurement_id_loop)).set_index(['EIS_Frequency'], append=True)
    return df


def filter_by_frequencies(df, frequencies, minimal_amount_of_frequencies):
    """
    Filter the dataframe by the frequencies.

    Parameters:
        df (pandas.DataFrame): The dataframe to filter.
        frequencies (list): The frequencies to filter by.
        minimal_amount_of_frequencies (int): The minimal amount of frequencies to filter by.

    Returns:
        pandas.DataFrame: The filtered dataframe.
    """

    measurements_before_delte = len(
        df.index.droplevel("EIS_Frequency").unique())

    pool = multiprocessing.Pool(cores)

    # split the dataframe into chunks for parallel processing, one chunk per core
    cell_measurement_list = df.index.droplevel("EIS_Frequency").unique()
    cell_names_list = np.array_split(cell_measurement_list, cores)
    df_new_index = df.reset_index(level=['EIS_Frequency'])
    df_list = [df_new_index.loc[cell_names_list[i]].set_index(
        ['EIS_Frequency'], append=True) for i in range(len(cell_names_list))]
    frequencies_list = list(itertools.repeat(frequencies, cores))
    minimal_amount_of_frequencies_list = list(
        itertools.repeat(minimal_amount_of_frequencies, cores))

    results = pool.starmap(filer_by_frequencies_groups_parallel_func, zip(
        df_list, frequencies_list, minimal_amount_of_frequencies_list))
    df = pd.concat(results)
    pool.close()

    measurements_after_delte = len(
        df.index.droplevel("EIS_Frequency").unique())

    sankey_dict.update(
        {'filter_by_frequencies': measurements_after_delte - measurements_before_delte})

    return df


def eis_interpolate_restruct_parallel_func(df, new_frequencies):
    """
    Given a dataframe and a list of new frequencies, interpolate the EIS dataframe to the new frequencies.

    Parameters:
        df (pd.DataFrame): The dataframe containing the EIS dataframe.
        new_frequencies (list): The list of new frequencies.

    Returns:
        pd.DataFrame: The interpolated dataframe.
    """
    if 'EIS_Z_phase' in df.columns:
        eis_column_names = ['EIS_Z_abs', 'EIS_Z_phase', 'EIS_Z_Re', 'EIS_Z_Im']
    else:
        eis_column_names = ['EIS_Z_abs']
    all_keys = list(df.keys())
    average_keys = list(
        filter(lambda key: key not in eis_column_names, all_keys))
    # set column names for new DataFrame
    eis_columns = average_keys + \
        get_eis_columns(eis_column_names, new_frequencies)
    # select individual measurements as index for new DataFrame
    measure_ids = df.index.droplevel(2)
    measure_ids = measure_ids[np.invert(measure_ids.duplicated(keep="first"))]
    # create empty new Dataframe
    df_new = pd.DataFrame(columns=eis_columns, index=measure_ids)

    for cell_ID_loop in df.index.unique("cell_ID"):
        for EIS_measurement_id_loop in df.loc[cell_ID_loop].index.unique("EIS_measurement_id"):
            frequency = df.loc[cell_ID_loop].loc[EIS_measurement_id_loop].index.to_numpy(
            )
            z_abs = df.loc[cell_ID_loop].loc[EIS_measurement_id_loop].loc[:, 'EIS_Z_abs'].to_numpy(
                dtype='float64')  # very important to use dtype='float64'!!!
            if 'EIS_Z_phase' in df.columns:
                z_phase = df.loc[cell_ID_loop].loc[EIS_measurement_id_loop].loc[:, 'EIS_Z_phase'].to_numpy(
                    dtype='float64')  # very important to use dtype='float64'!!!

            # print("Cell-ID: "+cell_ID_loop+" EIS ID: "+str(EIS_measurement_id_loop), flush=True)
            # bad error handling...
            if len(frequency) < 3:
                continue

            frequency, unique_ind = np.unique(frequency, return_index=True)
            z_abs = z_abs[unique_ind]
            if 'EIS_Z_phase' in df.columns:
                z_phase = z_phase[unique_ind]

            # interpolate
            abs_inter = interpolate.PchipInterpolator(frequency, z_abs)
            if 'EIS_Z_phase' in df.columns:
                phase_inter = interpolate.PchipInterpolator(frequency, z_phase)

            # evaluate
            z_abs_inter = abs_inter(new_frequencies)
            if 'EIS_Z_phase' in df.columns:
                z_phase_inter = phase_inter(new_frequencies)
                z_inter = z_abs_inter * np.exp(1j * z_phase_inter)
                z_real_inter = np.real(z_inter)
                z_imag_inter = np.imag(z_inter)
            if 'EIS_Z_phase' in df.columns:
                # critical to keep order correct!!!!
                z_dict = dict(
                    zip(eis_column_names, [z_abs_inter, z_phase_inter, z_real_inter, z_imag_inter]))
            else:
                # critical to keep order correct!!!!
                z_dict = dict(zip(eis_column_names, [z_abs_inter]))
            # put into the new df
            for f_index, f in enumerate(new_frequencies):
                for eis_key_index, eis_key in enumerate(eis_column_names):
                    column_key = get_column_key(eis_key, f)
                    df_new.loc[(cell_ID_loop, EIS_measurement_id_loop),
                               column_key] = z_dict[eis_column_names[eis_key_index]][f_index]

            for average_key_index, average_key in enumerate(average_keys):
                if pd.to_numeric(df.loc[cell_ID_loop].loc[EIS_measurement_id_loop].loc[:, average_key], errors='coerce').notnull().all():
                    value = np.mean(
                        df.loc[cell_ID_loop].loc[EIS_measurement_id_loop].loc[:, average_key].to_numpy(dtype='float64'))
                    df_new.loc[(cell_ID_loop, EIS_measurement_id_loop),
                               average_key] = value
                else:
                    if average_key == "Time":
                        value = np.min(df.loc[cell_ID_loop].loc[EIS_measurement_id_loop].loc[:, average_key].to_numpy(
                            dtype='datetime64[us]'))
                        df_new.loc[(cell_ID_loop, EIS_measurement_id_loop),
                                   average_key] = value
                    else:
                        value = df.loc[cell_ID_loop].loc[EIS_measurement_id_loop].loc[:,
                                                                                      average_key].values[0]
                        df_new.loc[(cell_ID_loop, EIS_measurement_id_loop),
                                   average_key] = value

    return df_new


def eis_interpolate_restruct(df, new_frequencies):
    """
    Given a dataframe and a list of new frequencies, interpolate the dataframe to the new frequencies.

    Parameters:
        df (pd.DataFrame): The dataframe to interpolate.
        new_frequencies (list): The new frequencies to interpolate to.

    Returns:
        pd.DataFrame: The interpolated dataframe.
    """
    pool = multiprocessing.Pool(cores)

    # split the dataframe into chunks for parallel processing, one chunk per core
    cell_measurement_list = df.index.droplevel("EIS_Frequency").unique()
    cell_names_list = np.array_split(cell_measurement_list, cores)
    df_new_index = df.reset_index(level=['EIS_Frequency'])
    df_list = [df_new_index.loc[cell_names_list[i]].set_index(
        ['EIS_Frequency'], append=True) for i in range(len(cell_names_list))]
    new_frequencies_list = list(itertools.repeat(new_frequencies, cores))

    results = pool.starmap(eis_interpolate_restruct_parallel_func, zip(
        df_list, new_frequencies_list))
    df = pd.concat(results)
    pool.close()
    return df


def add_DRT_parallel_func(df, key_lookup_df, drt_plot=False, feature=None, cmap=None):
    """
    Add the DRT values to the dataframe. If the DRT plot is enabled, plot the DRT values.
    Parameters:
        df (pd.DataFrame): The dataframe containing the DRT values.
        key_lookup_df (pd.DataFrame): The dataframe containing the key lookup.
        drt_plot (bool): Whether or not to plot the DRT values.
        feature (str): The feature to use for coloring the DRT plot.
        cmap (matplotlib.colors.Colormap): The colormap to use for coloring the DRT plot.
    Returns:
        pd.DataFrame or list: The modified dataframe with DRT values, or the list [fig, axes, df] if drt_plot is True.
    """

    if drt_plot:
        fig, axes = eisplot.plt.subplots()
        if eisplot.mpl.rcParams['text.usetex'] == True:
            eisplot.plt.xlabel(r"$\tau$ in s")
            eisplot.plt.ylabel(r'$\gamma$ in $\Omega$')
        else:
            eisplot.plt.xlabel('$\\tau$ in s')
            eisplot.plt.ylabel('$\gamma$ in $\Omega$')
        eisplot.plt.grid()

        if feature is not None:
            if cmap is None:
                cmap = eisplot.setup_colormap(
                    df[feature].min(), df[feature].max(), feature, fig, axes)
            colors = cmap.to_rgba(df[feature].to_numpy("float64"))

        hfig = IPdisplay.display(fig, display_id=True)

    DRT_dict = dict.fromkeys([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [])
    for ind, dic in enumerate(DRT_dict):
        DRT_dict[ind] = ["DRT_Peak_" +
                         str(dic) + "_tau", "DRT_Peak_" + str(dic) + "_gamma"]

    for cell_ID_loop in df.index.unique("cell_ID"):
        for EIS_measurement_id_loop in df.loc[cell_ID_loop].index.unique("EIS_measurement_id"):
            frequencies = key_lookup_df["frequency"].to_numpy()
            if len(frequencies) > 5:
                mask = (df.index.get_level_values(0) == cell_ID_loop) & (
                    df.index.get_level_values(1) == EIS_measurement_id_loop)

                abs_keys = key_lookup_df["EIS_Z_abs"].to_list()
                phase_keys = key_lookup_df["EIS_Z_phase"].to_list()

                abs_values = df.loc[mask, abs_keys].to_numpy(dtype='float64')[
                    0]
                phase_values = df.loc[mask, phase_keys].to_numpy(dtype='float64')[
                    0]
                impedance = abs_values * np.exp(1j * phase_values)

                N_freqs = len(frequencies)
                freq_min = np.min(frequencies)
                freq_max = np.max(frequencies)
                freq_vec = np.logspace(np.log10(freq_min), np.log10(
                    freq_max), num=N_freqs, endpoint=True)
                tau_vec = np.logspace(-np.log10(freq_max), -np.log10(
                    freq_min), num=N_freqs, endpoint=True)
                omega_vec = 2. * np.pi * freq_vec
                N_taus = tau_vec.shape[0]

                shape_control = 'FWHM'
                coeff = 0.5
                # expansion_type = 'PWL'
                expansion_type = 'Gaussian'
                # expansion_type = 'C0 Matern'
                # expansion_type = 'C2 Matern'
                # expansion_type = 'C4 Matern'
                # expansion_type = 'C6 Matern'
                # expansion_type = 'Inverse Quadratic'
                # expansion_type = 'Inverse Quadric'
                # expansion_type = 'Cauchy'
                data_used = 're+im'
                include_RL = 'L'  # 'R' 'R+L'
                derivative_RR = '1st'

                epsilon = general_fun.compute_epsilon(
                    freq_vec, coeff, expansion_type, shape_control)

                A_re, A_im, A = general_fun.assemble_A(
                    freq_vec, tau_vec, expansion_type, epsilon, include_RL, brute_force=True)
                M = general_fun.assemble_M(
                    tau_vec, expansion_type, epsilon, derivative_RR, include_RL)
                Z_re, Z_im, Z = general_fun.assemble_Z(impedance)

                log_lambda_0 = np.log(10 ** -3)
                cv_type = 'GCV'

                save_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                lambda_GCV = general_fun.optimal_lambda(
                    A_re, A_im, Z_re, Z_im, M, log_lambda_0, cv_type)
                sys.stdout = save_stdout

                if include_RL == 'R' or include_RL == 'L':
                    lb = np.zeros([N_freqs + 1])
                elif include_RL == 'R+L' or include_RL == 'L+R':
                    lb = np.zeros([N_freqs + 2])
                else:
                    lb = np.zeros([N_freqs])

                bound_mat = np.eye(lb.shape[0])

                save_stdout = sys.stdout
                sys.stdout = mystdout = StringIO()
                H_combined, c_combined = general_fun.quad_format_combined(
                    A_re, A_im, Z_re, Z_im, M, lambda_GCV)
                x_GCV = general_fun.cvxopt_solve_qpr(
                    H_combined, c_combined, -bound_mat, lb)
                sys.stdout = save_stdout

                if include_RL == 'R' or include_RL == 'L':
                    gamma_RR_GCV = general_fun.x_to_gamma(
                        x_GCV[1:], tau_vec, tau_vec, expansion_type, epsilon)
                elif include_RL == 'R+L' or include_RL == 'L+R':
                    gamma_RR_GCV = general_fun.x_to_gamma(
                        x_GCV[2:], tau_vec, tau_vec, expansion_type, epsilon)
                else:
                    gamma_RR_GCV = general_fun.x_to_gamma(
                        x_GCV[:], tau_vec, tau_vec, expansion_type, epsilon)

                peaks, properties = find_peaks(gamma_RR_GCV)

                for ind, value in enumerate(DRT_dict):
                    if ind >= len(peaks):
                        df.loc[mask, DRT_dict[value][0]] = np.nan
                        df.loc[mask, DRT_dict[value][1]] = np.nan
                    else:
                        df.loc[mask, DRT_dict[value][0]] = tau_vec[peaks][ind]
                        df.loc[mask, DRT_dict[value][1]
                               ] = gamma_RR_GCV[peaks][ind]

                if drt_plot:
                    if feature is not None:
                        axes.semilogx(tau_vec, gamma_RR_GCV,
                                      c=colors[mask], linestyle='-')
                        axes.plot(tau_vec[peaks], gamma_RR_GCV[peaks],
                                  "x", c=colors[mask], linestyle='None', alpha=1)
                    else:
                        axes.semilogx(tau_vec, gamma_RR_GCV)
                        axes.plot(tau_vec[peaks], gamma_RR_GCV[peaks],
                                  "x", linestyle='None', alpha=1)
                    hfig.update(fig)
                    time.sleep(0.1)
    if drt_plot:
        return [fig, axes, df]
    else:
        return df


def add_DRT(df, key_lookup_df, drt_plot=False, feature=None, cmap=None):
    """
    Add DRT to the dataframe.

    Parameters:
        df (pd.DataFrame): The dataframe to add DRT to.
        key_lookup_df (pd.DataFrame): The key lookup dataframe.
        drt_plot (bool): Whether or not to plot the DRT.
        feature (str): The feature to use for coloring the DRT plot.
        cmap (matplotlib.colors.Colormap): The colormap to use for coloring the DRT plot.

    Returns:
        pd.DataFrame or list: The modified dataframe with DRT values, or the list [fig, axes, df] if drt_plot is True.
    """

    if drt_plot:
        cpu_count = 1
    else:
        cpu_count = cores
        pool = multiprocessing.Pool(cores)

    cell_measurement_list = df.index.unique()
    cell_names_list = np.array_split(cell_measurement_list, cpu_count)
    df_list = list(df.loc[cell_names_list[i]]
                   for i in range(len(cell_names_list)))
    key_lookup_df_list = list(itertools.repeat(key_lookup_df, cpu_count))
    drt_plot_list = list(itertools.repeat(drt_plot, cpu_count))

    if drt_plot:
        [fig, axes, df] = add_DRT_parallel_func(
            df, key_lookup_df, drt_plot, feature, cmap)
    else:
        results = pool.starmap(add_DRT_parallel_func, zip(
            df_list, key_lookup_df_list, drt_plot_list))
        df = pd.concat(results)
        pool.close()

    if drt_plot:
        return [fig, axes, df]
    else:
        return df


def add_EIS_Extrema_parallel_func(df, key_lookup_df):
    """
    Add the EIS extrema to the dataframe. This is done in parallel.

    Parameters:
        df (pd.DataFrame): The dataframe containing the EIS data.
        key_lookup_df (pd.DataFrame): The dataframe containing the key lookup data.

    Returns:
        pd.DataFrame: The dataframe with the EIS extrema added.
    """
    EIS_Extrema_dict = dict.fromkeys([0, 1, 2], [])
    for ind, dic in enumerate(EIS_Extrema_dict):
        EIS_Extrema_dict[ind] = [
            "Nyquist_Zero_Real_Freq_" + str(dic),
            "Nyquist_Zero_Real_Value_" + str(dic),
            "Nyquist_Zero_Imag_Freq_" + str(dic),
            "Nyquist_Zero_Imag_Value_" + str(dic),
            "Nyquist_Min_Real_Freq_" + str(dic),
            "Nyquist_Min_Real_Value_" + str(dic),
            "Nyquist_Min_Imag_Freq_" + str(dic),
            "Nyquist_Min_Imag_Value_" + str(dic),
            "Nyquist_Max_Real_Freq_" + str(dic),
            "Nyquist_Max_Real_Value_" + str(dic),
            "Nyquist_Max_Imag_Freq_" + str(dic),
            "Nyquist_Max_Imag_Value_" + str(dic),
            "Bode_Abs_Min_Freq_" + str(dic),
            "Bode_Abs_Min_Value_" + str(dic),
            "Bode_Abs_Max_Freq_" + str(dic),
            "Bode_Abs_Max_Value_" + str(dic),
            "Bode_Phase_Min_Freq_" + str(dic),
            "Bode_Phase_Min_Value_" + str(dic),
            "Bode_Phase_Max_Freq_" + str(dic),
            "Bode_Phase_Max_Value_" + str(dic)
        ]

    for cell_ID_loop in df.index.unique("cell_ID"):
        for EIS_measurement_id_loop in df.loc[cell_ID_loop].index.unique("EIS_measurement_id"):
            frequencies = key_lookup_df["frequency"].to_numpy(dtype='float64')
            if len(frequencies) > 5:  # bad error handling
                mask = (df.index.get_level_values(0) == cell_ID_loop) & (
                    df.index.get_level_values(1) == EIS_measurement_id_loop)

                abs_keys = key_lookup_df["EIS_Z_abs"].to_list()
                phase_keys = key_lookup_df["EIS_Z_phase"].to_list()

                abs_values = df.loc[mask, abs_keys].to_numpy(dtype='float64')[
                    0]
                phase_values = df.loc[mask, phase_keys].to_numpy(dtype='float64')[
                    0]
                impedance = abs_values * np.exp(1j * phase_values)

                real_values = np.real(impedance)
                imag_values = np.imag(impedance)

                # find peaks
                nyquist_zero, properties = find_peaks(-abs(imag_values))
                nyquist_min, properties = find_peaks(imag_values)
                nyquist_max, properties = find_peaks(-imag_values)
                bode_abs_min, properties = find_peaks(-abs_values)
                bode_abs_max, properties = find_peaks(abs_values)
                bode_phase_min, properties = find_peaks(-phase_values)
                bode_phase_max, properties = find_peaks(phase_values)

                # if no data is available for the selected EIS_Extrema_dict, fill with NaN's
                for ind, value in enumerate(EIS_Extrema_dict):
                    if ind >= len(nyquist_zero):
                        df.loc[mask, EIS_Extrema_dict[value][0:4]] = np.nan
                    else:
                        df.loc[mask, EIS_Extrema_dict[value][0:4]] = [
                            frequencies[nyquist_zero[ind]], real_values[nyquist_zero[ind]], frequencies[nyquist_zero[ind]], imag_values[nyquist_zero[ind]]]

                    if ind >= len(nyquist_min):
                        df.loc[mask, EIS_Extrema_dict[value][4:8]] = np.nan
                    else:
                        df.loc[mask, EIS_Extrema_dict[value][4:8]] = [
                            frequencies[nyquist_min[ind]], real_values[nyquist_min[ind]], frequencies[nyquist_min[ind]], imag_values[nyquist_min[ind]]]

                    if ind >= len(nyquist_max):
                        df.loc[mask, EIS_Extrema_dict[value][8:12]] = np.nan
                    else:
                        df.loc[mask, EIS_Extrema_dict[value][8:12]] = [
                            frequencies[nyquist_max[ind]], real_values[nyquist_max[ind]], frequencies[nyquist_max[ind]], imag_values[nyquist_max[ind]]]

                    if ind >= len(bode_abs_min):
                        df.loc[mask, EIS_Extrema_dict[value][12:14]] = np.nan
                    else:
                        df.loc[mask, EIS_Extrema_dict[value][12:14]] = [
                            frequencies[bode_abs_min[ind]], abs_values[bode_abs_min[ind]]]

                    if ind >= len(bode_abs_max):
                        df.loc[mask, EIS_Extrema_dict[value][14:16]] = np.nan
                    else:
                        df.loc[mask, EIS_Extrema_dict[value][14:16]] = [
                            frequencies[bode_abs_max[ind]], abs_values[bode_abs_max[ind]]]

                    if ind >= len(bode_phase_min):
                        df.loc[mask, EIS_Extrema_dict[value][16:18]] = np.nan
                    else:
                        df.loc[mask, EIS_Extrema_dict[value][16:18]] = [
                            frequencies[bode_phase_min[ind]], phase_values[bode_phase_min[ind]]]

                    if ind >= len(bode_phase_max):
                        df.loc[mask, EIS_Extrema_dict[value][18:20]] = np.nan
                    else:
                        df.loc[mask, EIS_Extrema_dict[value][18:20]] = [
                            frequencies[bode_phase_max[ind]], phase_values[bode_phase_max[ind]]]

    return df


def add_EIS_Extrema(df, key_lookup_df):
    """
    Add the EIS extrema to the dataframe. This is done in parallel.

    Parameters:
        df (pd.DataFrame): The dataframe containing the EIS data.
        key_lookup_df (pd.DataFrame): The dataframe containing the key lookup data.

    Returns:
        pd.DataFrame: The dataframe with the EIS extrema added.
    """
    pool = multiprocessing.Pool(cores)

    cell_measurement_list = df.index.unique()
    cell_names_list = np.array_split(cell_measurement_list, cores)
    df_list = [df.loc[cell_names_list[i]] for i in range(len(cell_names_list))]
    key_lookup_df_list = list(itertools.repeat(key_lookup_df, cores))

    results = pool.starmap(add_EIS_Extrema_parallel_func,
                           zip(df_list, key_lookup_df_list))
    df = pd.concat(results)
    pool.close()
    return df
