# -*- coding: utf-8 -*-

import json
import os
import numpy as np
import pandas as pd
from modules import eisplot as eisplot
from modules.eisplot import plt
from modules.eisplot import mpl
import mlflow
from sklearn.model_selection import train_test_split

# Constants for scaling the data
use_arrhenius_correction = True
use_arrhenius_correction_with_factor = False

use_min_max_scaler = True
use_standard_scaler = False
scale_y_data = False

arrhenius_b = -15.47
arrhenius_c = 1.30

x_min = np.array(0, dtype=np.float32)
x_max = np.array(0, dtype=np.float32)
x_mean = np.array(0, dtype=np.float32)
x_std = np.array(0, dtype=np.float32)

y_min = np.array(0, dtype=np.float32)
y_max = np.array(0, dtype=np.float32)
y_mean = np.array(0, dtype=np.float32)
y_std = np.array(0, dtype=np.float32)

cm = 1 / 2.54  # centimeters in inches


def evaluate_max_abs_error(model, x, y):
    """
    Evaluate the Maximum absolute error of a model's predictions.

    Parameters:
        model (object): The model used for making predictions. It must have a `predict` method.
        x (array-like): The input data for making predictions.
        y (array-like): The true target values.

    Returns:
        float: Maximum absolute error of a model's predictions.

    Notes:
        - If `scale_y_data` is True, the function will inverse transform the predictions and true values
        using either Min-Max scaling or Standard scaling, depending on the flags `use_min_max_scaler`
        and `use_standard_scaler`.
        - The variables `scale_y_data`, `use_min_max_scaler`, `use_standard_scaler`, `y_min`, `y_max`,
        `y_mean`, and `y_std` are assumed to be defined in the global scope.
    """

    y_pred = model.predict(x)
    if scale_y_data:
        if use_min_max_scaler:
            y_pred = inverse_min_max_scaler(y_pred, y_min, y_max).ravel()
            y_orig = inverse_min_max_scaler(y, y_min, y_max).ravel()
        elif use_standard_scaler:
            y_pred = inverse_standard_scaler(y_pred, y_mean, y_std).ravel()
            y_orig = inverse_standard_scaler(y, y_mean, y_std).ravel()
    else:
        y_pred = y_pred.ravel()
        y_orig = y.ravel()
    max_abs_error = np.max(np.mean(y_pred - y_orig))
    return max_abs_error


def evaluate_mse(model, x, y):
    """
    Evaluate the Mean Squared Error (MSE) of a model's predictions.

    Parameters:
        model (object): The model used for making predictions. It must have a `predict` method.
        x (array-like): The input data for making predictions.
        y (array-like): The true target values.

    Returns:
        float: The Mean Squared Error (MSE) of the model's predictions.

    Notes:
        - If `scale_y_data` is True, the function will inverse transform the predictions and true values
        using either Min-Max scaling or Standard scaling, depending on the flags `use_min_max_scaler`
        and `use_standard_scaler`.
        - The variables `scale_y_data`, `use_min_max_scaler`, `use_standard_scaler`, `y_min`, `y_max`,
        `y_mean`, and `y_std` are assumed to be defined in the global scope.
    """

    y_pred = model.predict(x)
    if scale_y_data:
        if use_min_max_scaler:
            y_pred = inverse_min_max_scaler(y_pred, y_min, y_max).ravel()
            y_orig = inverse_min_max_scaler(y, y_min, y_max).ravel()
        elif use_standard_scaler:
            y_pred = inverse_standard_scaler(y_pred, y_mean, y_std).ravel()
            y_orig = inverse_standard_scaler(y, y_mean, y_std).ravel()
    else:
        y_pred = y_pred.ravel()
        y_orig = y.ravel()
    mse = np.mean((y_pred - y_orig) ** 2)
    return mse


def evaluate_rmse(model, x, y):
    """
    Evaluate the Root Mean Squared Error (RMSE) of a given model.

    Parameters:
        model : object
            The model to evaluate. It should have a predict method.
        x : array-like
            The input data to predict.
        y : array-like
            The true values corresponding to the input data.

    Returns:
        float: he RMSE value.
    """

    rmse = np.sqrt(evaluate_mse(model, x, y))
    return rmse


def min_max_scaler(values, min, max):
    """
    Scales the given values to a range between 0 and 1 using min-max normalization.

    Parameters:
        values (array-like): The values to be scaled.
        min (float): The minimum value of the original range.
        max (float): The maximum value of the original range.

    Returns:
        array-like: The scaled values.
    """
    values_scaled = (values - min) / (max - min)
    return values_scaled


def inverse_min_max_scaler(values, min, max):
    """
    Reverts the min-max scaling transformation on a given set of values.

    Parameters:
        values (array-like): The scaled values to be transformed back.
        min (float): The minimum value used in the original scaling.
        max (float): The maximum value used in the original scaling.

    Returns:
        array-like: The values transformed back to their original scale.
    """
    inverse_values = values * (max - min) + min
    return inverse_values


def standard_scaler(values, mean, standard_diviation):
    """
    Scales the input values using the standard scaling method.

    Parameters:
        values (array-like): The data to be scaled.
        mean (float): The mean value of the data.
        standard_diviation (float): The standard deviation of the data.

    Returns:
        array-like: The scaled values.
    """
    values_scaled = (values - mean) / standard_diviation
    return values_scaled


def inverse_standard_scaler(values, mean, standard_diviation):
    """
    Reverts the standard scaling transformation on a set of values.
    This function takes a set of values that have been standardized (i.e.,
    transformed to have a mean of 0 and a standard deviation of 1) and
    reverts them back to their original scale using the provided mean and
    standard deviation.

    Parameters:
        values (array-like): The standardized values to be transformed back
                             to the original scale.
        mean (float): The mean of the original data before standard scaling.
        standard_diviation (float): The standard deviation of the original
                                    data before standard scaling.

    Returns:
        array-like: The values transformed back to the original scale.
    """

    inverse_values = values * standard_diviation + mean
    return inverse_values


def arrhenius_correction(value):
    """
    Apply Arrhenius correction to a given value.

    Parameters:
        value (float): The value to be corrected.

    Returns:
        float: The corrected value.
    """
    if use_arrhenius_correction_with_factor:
        return arrhenius_b * np.log(arrhenius_c * value)
    else:
        return np.log(1 / value)


def arrhenius_correction_inverse(value):
    """
    Applies an inverse Arrhenius correction to the given value.

    Parameters:
        value (float): The value to be corrected.

    Returns:
        float: The corrected value.
    """
    if use_arrhenius_correction_with_factor:
        return 1 / arrhenius_c * np.exp(value / arrhenius_b)
    else:
        return 1 / np.exp(value)


def plot_diag_during_fitting_mimo(
    model,
    x_train,
    x_validation,
    x_test,
    y_train,
    y_validation,
    y_test,
    data_set,
    output_parameters_name,
    merged_params,
):
    """
    Plots diagnostic graphs during model fitting for a Multiple Input Multiple Output (MIMO) system.

    Parameters:
        model : object
            The machine learning model used for predictions.
        x_train : array-like
            The training set features.
        x_validation : array-like
            The validation set features.
        x_test : array-like
            The test set features.
        y_train : array-like
            The training set output parameters.
        y_validation : array-like
            The validation set output parameters.
        y_test : array-like
            The test set output parameters.
        data_set : dict
            A dictionary containing the training, validation, and test datasets.
        output_parameters_name : str
            The name of the output parameters.
        merged_params : dict
            A dictionary of merged parameters for logging and plotting.

    Returns:
        None
    """
    y_pred_train = model.predict(x_train)
    y_pred_validation = model.predict(x_validation)
    y_pred_test = model.predict(x_test)
    if scale_y_data:
        if use_min_max_scaler:
            y_pred_train = inverse_min_max_scaler(y_pred_train, y_min, y_max)
            y_pred_validation = inverse_min_max_scaler(y_pred_validation, y_min, y_max)
            y_pred_test = inverse_min_max_scaler(y_pred_test, y_min, y_max)
            y_train_plot = inverse_min_max_scaler(y_train, y_min, y_max)
            y_validation_plot = inverse_min_max_scaler(y_validation, y_min, y_max)
            y_test_plot = inverse_min_max_scaler(y_test, y_min, y_max)
        elif use_standard_scaler:
            y_pred_train = inverse_standard_scaler(y_pred_train, y_mean, y_std)
            y_pred_validation = inverse_standard_scaler(
                y_pred_validation, y_mean, y_std
            )
            y_pred_test = inverse_standard_scaler(y_pred_test, y_mean, y_std)
            y_train_plot = inverse_standard_scaler(y_train, y_mean, y_std)
            y_validation_plot = inverse_standard_scaler(y_validation, y_mean, y_std)
            y_test_plot = inverse_standard_scaler(y_test, y_mean, y_std)
    else:
        y_train_plot = y_train
        y_validation_plot = y_validation
        y_test_plot = y_test
    fig, ax = plt.subplots(1, 1, figsize=(7 * cm, 7 * cm), layout="compressed")
    plt.cla()
    eisplot.setup_scatter(
        data_set,
        0.0,
        title=False,
        legend=False,
        fig=fig,
        ax=ax,
        ax_xlabel=True,
        ax_ylabel=True,
        subplots_adjust=False,
        add_trendline=True,
        label=output_parameters_name,
    )
    ax.plot(
        y_train_plot,
        y_pred_train,
        ".",
        color=eisplot.rwth_colors.colors[("petrol", 100)],
        alpha=0.5,
    )
    ax.plot(
        y_validation_plot,
        y_pred_validation,
        "1",
        color=eisplot.rwth_colors.colors[("turqoise", 100)],
        alpha=0.5,
    )
    ax.plot(
        y_test_plot,
        y_pred_test,
        "2",
        color=eisplot.rwth_colors.colors[("blue", 100)],
        alpha=0.5,
    )
    legend_elements = [
        mpl.lines.Line2D(
            [0], [0], color=eisplot.rwth_colors.colors[("green", 100)], label="ideal"
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            marker=".",
            color=eisplot.rwth_colors.colors[("petrol", 100)],
            linestyle="",
            label="train",
            alpha=0.5,
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            marker="1",
            color=eisplot.rwth_colors.colors[("turqoise", 100)],
            linestyle="",
            label="validation",
            alpha=0.5,
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            marker="2",
            color=eisplot.rwth_colors.colors[("blue", 100)],
            linestyle="",
            label="test",
            alpha=0.5,
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="best",
        scatterpoints=1,
        prop={"size": eisplot.font_size},
    )
    mlflow.log_figure(fig, "prediction_vs_actual." + merged_params["log_plot_type"])
    plt.close()


def plot_diag_during_fitting(
    model,
    name_of_this_run,
    output_parameter,
    x_test,
    x_train,
    x_validation,
    data_set,
    train_rmse_temp,
    validation_rmse_temp,
    test_rmse_temp,
    merged_params,
    axes_labels=True,
):
    """
    Plots diagnostic graphs during model fitting.

    Parameters:
        model : object
            The machine learning model used for predictions.
        name_of_this_run : str
            The name of the current run.
        output_parameter : str
            The output parameter being predicted.
        x_test : array-like
            The test set features.
        x_train : array-like
            The training set features.
        x_validation : array-like
            The validation set features.
        data_set : dict
            A dictionary containing the training, validation, and test datasets.
        train_rmse_temp : float
            The RMSE of the training set.
        validation_rmse_temp : float
            The RMSE of the validation set.
        test_rmse_temp : float
            The RMSE of the test set.
        merged_params : dict
            A dictionary of merged parameters for logging and plotting.
        axes_labels : bool, optional

    Returns:
        None
    """

    fig, ax = plt.subplots(1, 1, figsize=(7 * cm, 7 * cm))
    plt.cla()

    # prediction on train set
    y_pred = model.predict(x_train)
    if scale_y_data:
        if use_min_max_scaler:
            y_pred = inverse_min_max_scaler(y_pred, y_min, y_max).ravel()
        elif use_standard_scaler:
            y_pred = inverse_standard_scaler(y_pred, y_mean, y_std).ravel()
    cell_list = list(set(data_set["df_train"].index.get_level_values(0)))
    fig, ax = eisplot.cell_scatter(
        data_set,
        y_pred.ravel(),
        cell_names=cell_list,
        title=False,
        legend=False,
        fig=fig,
        ax=ax,
        ax_xlabel=axes_labels,
        ax_ylabel=axes_labels,
    )

    # prediction on validation set
    y_pred = model.predict(x_validation)
    if scale_y_data:
        if use_min_max_scaler:
            y_pred = inverse_min_max_scaler(y_pred, y_min, y_max).ravel()
        elif use_standard_scaler:
            y_pred = inverse_standard_scaler(y_pred, y_mean, y_std).ravel()
    cell_list = list(set(data_set["df_validation"].index.get_level_values(0)))
    fig, ax = eisplot.cell_scatter(
        data_set,
        y_pred,
        is_validation=True,
        cell_names=cell_list,
        title=False,
        legend=False,
        fig=fig,
        ax=ax,
        add_trendline=False,
        ax_xlabel=axes_labels,
        ax_ylabel=axes_labels,
    )

    # prediction on test set
    y_pred = model.predict(x_test)
    if scale_y_data:
        if use_min_max_scaler:
            y_pred = inverse_min_max_scaler(y_pred, y_min, y_max).ravel()
        elif use_standard_scaler:
            y_pred = inverse_standard_scaler(y_pred, y_mean, y_std).ravel()
    cell_list = list(set(data_set["df_test"].index.get_level_values(0)))
    fig, ax = eisplot.cell_scatter(
        data_set,
        y_pred,
        is_test=True,
        cell_names=cell_list,
        title=False,
        legend=False,
        fig=fig,
        ax=ax,
        add_trendline=False,
        ax_xlabel=axes_labels,
        ax_ylabel=axes_labels,
    )

    if (name_of_this_run == "example_data") & (output_parameter == "Temperature"):
        ax.set_xlim([-30, 60])
        ax.set_ylim([-30, 60])

        ax.text(
            -4,
            -19,
            "Train RMSE: " + "%.2f" % train_rmse_temp + " K",
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=eisplot.font_size,
        )
        ax.text(
            -4,
            -23,
            "Validation RMSE: " + "%.2f" % validation_rmse_temp + " K",
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=eisplot.font_size,
        )
        ax.text(
            -4,
            -27,
            "Test RMSE: " + "%.2f" % test_rmse_temp + " K",
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=eisplot.font_size,
        )

    legend_elements = [
        mpl.lines.Line2D(
            [0], [0], color=eisplot.rwth_colors.colors[("green", 100)], label="ideal"
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            marker=".",
            color=eisplot.rwth_colors.colors[("petrol", 100)],
            linestyle="",
            label="train",
            alpha=0.5,
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            marker="1",
            color=eisplot.rwth_colors.colors[("turqoise", 100)],
            linestyle="",
            label="validation",
            alpha=0.5,
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            marker="2",
            color=eisplot.rwth_colors.colors[("blue", 100)],
            linestyle="",
            label="test",
            alpha=0.5,
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="best",
        scatterpoints=1,
        prop={"size": eisplot.font_size},
    )
    fig.subplots_adjust(bottom=0.14, left=0.19)
    mlflow.log_figure(fig, "prediction_vs_actual." + merged_params["log_plot_type"])
    plt.close()


def plot_fit_during_fitting(
    model,
    name_of_this_run,
    name_of_the_feature,
    output_parameter,
    x_train,
    x_validation,
    x_test,
    y_train,
    y_validation,
    y_test,
    train_rmse_temp,
    validation_rmse_temp,
    test_rmse_temp,
    merged_params,
):
    """
    Plots the fit of a model during the fitting process for a Single Input Single Output (SISO) system.

    Parameters:
        model : object
            The trained model used for prediction.
        name_of_this_run : str
            The name of the current run.
        name_of_the_feature : str
            The name of the feature being used.
        output_parameter : str
            The output parameter being predicted.
        x_train : array-like
            Training data for the input feature.
        x_validation : array-like
            Validation data for the input feature.
        x_test : array-like
            Test data for the input feature.
        y_train : array-like
            Training data for the output parameter.
        y_validation : array-like
            Validation data for the output parameter.
        y_test : array-like
            Test data for the output parameter.
        train_rmse_temp : float
            Root Mean Square Error (RMSE) for the training data.
        validation_rmse_temp : float
            Root Mean Square Error (RMSE) for the validation data.
        test_rmse_temp : float
            Root Mean Square Error (RMSE) for the test data.
        merged_params : dict
            Dictionary containing additional parameters for logging and plotting.

    Returns:
        None
    """

    fig, ax = plt.subplots(1, 1, figsize=(7 * cm, 7 * cm))
    plt.cla()

    if use_min_max_scaler:
        x_min_plot = inverse_min_max_scaler(
            np.min(np.concatenate((x_train, x_validation, x_test))), x_min, x_max
        )
        x_max_plot = inverse_min_max_scaler(
            np.max(np.concatenate((x_train, x_validation, x_test))), x_min, x_max
        )
    elif use_standard_scaler:
        x_min_plot = inverse_standard_scaler(
            np.min(np.concatenate((x_train, x_validation, x_test))), x_mean, x_std
        )
        x_max_plot = inverse_standard_scaler(
            np.max(np.concatenate((x_train, x_validation, x_test))), x_mean, x_std
        )
    else:
        x_min_plot = np.min(np.concatenate((x_train, x_validation, x_test)))
        x_max_plot = np.max(np.concatenate((x_train, x_validation, x_test)))

    if use_arrhenius_correction:
        x_tmp = x_min_plot
        x_min_plot = arrhenius_correction_inverse(x_max_plot)
        x_max_plot = arrhenius_correction_inverse(x_tmp)

    x_min_plot = x_min_plot * 0.8
    x_max_plot = x_max_plot * 1.2

    x_plot = np.linspace(x_min_plot, x_max_plot, 1000, dtype=np.float32)[:, None]
    # x_plot= np.linspace(0,np.max(x)+np.max(x),1000,dtype=np.float32)[:, None]

    x_plot_arrhenius = x_plot
    x_train_arrhenius = x_train
    x_validation_arrhenius = x_validation
    x_test_arrhenius = x_test

    if use_arrhenius_correction:
        x_plot = arrhenius_correction(x_plot)

    if use_min_max_scaler:
        x_plot = min_max_scaler(x_plot, x_min, x_max)
        x_train_arrhenius = inverse_min_max_scaler(x_train_arrhenius, x_min, x_max)
        x_validation_arrhenius = inverse_min_max_scaler(
            x_validation_arrhenius, x_min, x_max
        )
        x_test_arrhenius = inverse_min_max_scaler(x_test_arrhenius, x_min, x_max)
    elif use_standard_scaler:
        x_plot = standard_scaler(x_plot, x_mean, x_std)
        x_train_arrhenius = inverse_standard_scaler(x_train_arrhenius, x_mean, x_std)
        x_validation_arrhenius = inverse_standard_scaler(
            x_validation_arrhenius, x_mean, x_std
        )
        x_test_arrhenius = inverse_standard_scaler(x_test_arrhenius, x_mean, x_std)

    if use_arrhenius_correction:
        x_train_arrhenius = arrhenius_correction_inverse(x_train_arrhenius)
        x_validation_arrhenius = arrhenius_correction_inverse(x_validation_arrhenius)
        x_test_arrhenius = arrhenius_correction_inverse(x_test_arrhenius)

    y_predicted = model.predict(x_plot)

    y_train_plot = y_train
    y_validation_plot = y_validation
    y_test_plot = y_test

    if scale_y_data:
        if use_min_max_scaler:
            y_predicted = inverse_min_max_scaler(y_predicted, y_min, y_max).ravel()
            y_train_plot = inverse_min_max_scaler(y_train_plot, y_min, y_max).ravel()
            y_validation_plot = inverse_min_max_scaler(
                y_validation_plot, y_min, y_max
            ).ravel()
            y_test_plot = inverse_min_max_scaler(y_test_plot, y_min, y_max).ravel()
        elif use_standard_scaler:
            y_predicted = inverse_standard_scaler(y_predicted, y_mean, y_std).ravel()
            y_train_plot = inverse_standard_scaler(y_train_plot, y_mean, y_std).ravel()
            y_validation_plot = inverse_standard_scaler(
                y_validation_plot, y_mean, y_std
            ).ravel()
            y_test_plot = inverse_standard_scaler(y_test_plot, y_mean, y_std).ravel()

    ax.plot(
        x_plot_arrhenius,
        y_predicted,
        lw=2,
        label="SVR",
        color=eisplot.rwth_colors.colors[("bordeaux", 100)],
    )
    ax.scatter(
        x_train_arrhenius,
        y_train_plot,
        marker=".",
        label="train",
        color=eisplot.rwth_colors.colors[("petrol", 100)],
        alpha=0.5,
    )
    ax.scatter(
        x_validation_arrhenius,
        y_validation_plot,
        marker="1",
        label="validation",
        color=eisplot.rwth_colors.colors[("turqoise", 100)],
        alpha=0.5,
    )
    ax.scatter(
        x_test_arrhenius,
        y_test_plot,
        marker="2",
        label="test",
        color=eisplot.rwth_colors.colors[("blue", 100)],
        alpha=0.5,
    )

    ax.set_ylabel("Temperature in $^\circ$C")
    if eisplot.mpl.rcParams["text.usetex"]:
        ax.set_xlabel(r"$|\underline{Z}|$ in m$\Omega$")
    else:
        ax.set_xlabel(r"|Z| in mΩ")
    ax.grid()

    if (name_of_this_run == "example_data") & (output_parameter == "Temperature"):
        ax.set_ylim([-30, 60])

        if name_of_the_feature == "_abs_0-01hz":
            ax.text(
                2,
                19,
                "Train RMSE: " + "%.2f" % train_rmse_temp + " K",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=eisplot.font_size,
            )
            ax.text(
                2,
                15,
                "Validation RMSE: " + "%.2f" % validation_rmse_temp + " K",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=eisplot.font_size,
            )
            ax.text(
                2,
                11,
                "Test RMSE: " + "%.2f" % test_rmse_temp + " K",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=eisplot.font_size,
            )
        elif name_of_the_feature == "_abs_0-1hz":
            ax.text(
                0,
                -19,
                "Train RMSE: " + "%.2f" % train_rmse_temp + " K",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=eisplot.font_size,
            )
            ax.text(
                0,
                -23,
                "Validation RMSE: " + "%.2f" % validation_rmse_temp + " K",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=eisplot.font_size,
            )
            ax.text(
                0,
                -27,
                "Test RMSE: " + "%.2f" % test_rmse_temp + " K",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=eisplot.font_size,
            )
        elif name_of_the_feature == "_abs_1hz":
            ax.text(
                0.05,
                -19,
                "Train RMSE: " + "%.2f" % train_rmse_temp + " K",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=eisplot.font_size,
            )
            ax.text(
                0.05,
                -23,
                "Validation RMSE: " + "%.2f" % validation_rmse_temp + " K",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=eisplot.font_size,
            )
            ax.text(
                0.05,
                -27,
                "Test RMSE: " + "%.2f" % test_rmse_temp + " K",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=eisplot.font_size,
            )
        elif name_of_the_feature == "_abs_10hz":
            ax.text(
                0.05,
                -19,
                "Train RMSE: " + "%.2f" % train_rmse_temp + " K",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=eisplot.font_size,
            )
            ax.text(
                0.05,
                -23,
                "Validation RMSE: " + "%.2f" % validation_rmse_temp + " K",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=eisplot.font_size,
            )
            ax.text(
                0.05,
                -27,
                "Test RMSE: " + "%.2f" % test_rmse_temp + " K",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=eisplot.font_size,
            )
        elif name_of_the_feature == "_abs_100hz":
            ax.text(
                0.025,
                -19,
                "Train RMSE: " + "%.2f" % train_rmse_temp + " K",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=eisplot.font_size,
            )
            ax.text(
                0.025,
                -23,
                "Validation RMSE: " + "%.2f" % validation_rmse_temp + " K",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=eisplot.font_size,
            )
            ax.text(
                0.025,
                -27,
                "Test RMSE: " + "%.2f" % test_rmse_temp + " K",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=eisplot.font_size,
            )
        elif name_of_the_feature == "_abs_1khz":
            ax.text(
                0.03,
                -19,
                "Train RMSE: " + "%.2f" % train_rmse_temp + " K",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=eisplot.font_size,
            )
            ax.text(
                0.03,
                -23,
                "Validation RMSE: " + "%.2f" % validation_rmse_temp + " K",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=eisplot.font_size,
            )
            ax.text(
                0.03,
                -27,
                "Test RMSE: " + "%.2f" % test_rmse_temp + " K",
                horizontalalignment="left",
                verticalalignment="center",
                fontsize=eisplot.font_size,
            )

    ax.legend(loc="best", scatterpoints=1, prop={"size": eisplot.font_size})

    fig.subplots_adjust(bottom=0.14, left=0.19)

    mlflow.log_figure(fig, "svr_fit." + merged_params["log_plot_type"])
    plt.close()


def python_matrix_to_c_matrix(matrix, matrix_name, max_rows=100):
    """
    Converts a Python matrix to a C-style matrix initialization string.

    Parameters:
        matrix (array-like): The Python matrix to be converted.
        matrix_name (str): The name of the C-style matrix.
        max_rows (int, optional): The maximum number of rows to include in the C-style matrix.

    Returns:
        str: The C-style matrix initialization string.
    """
    c_matrix = f"float {matrix_name}[{max_rows}][{len(matrix[0])}] = {{\n"
    for idx, row in enumerate(matrix):
        if idx >= max_rows:
            break
        c_matrix += "    {" + ", ".join(map(str, row)) + "},\t\n"
    c_matrix += "};"
    return c_matrix


def create_test_header_file(
    data_set_eval,
    header_path="microcontroller_eis_network/Core/Inc/",
    unique_model_name="example_model_name",
    polynomial_degree=1,
    max_rows=100
):
    """
    Generates a C header file containing test data arrays and scaler parameters.

    Parameters:
        data_set_eval (dict): A dictionary containing the test data arrays.
                                Expected keys are "test" with values being tuples of
                                (input_data, output_data).
        header_path (str, optional): The directory path where the header file will be saved.
                                        Defaults to "microcontroller_eis_network/Core/Inc/".
        unique_model_name (str, optional): A unique name for the model. Defaults to "example_model_name".
        polynomial_degree (int, optional): The degree of the polynomial used for feature engineering.
                                            Defaults to 1.
        max_rows (int, optional): The maximum number of rows to include in the C-style matrix.

    Returns:
        None

    The function performs the following steps:
        1. Extracts the test input and output data from the provided dataset.
        2. Converts the Python arrays to C-style arrays.
        3. Constructs a string containing the scaler parameters and their values.
        4. Combines the header, scaler parameters, test data arrays, and footer into a single string.
        5. Writes the combined string to a header file named "test_arrays.h" in the specified directory.
    """

    if len(data_set_eval["test"][0]) > max_rows:
        print("Warning: Too many test values. Randomly selecting {max_rows}.".format(max_rows=max_rows))
        random_indices = np.random.choice(len(data_set_eval["test"][0]), len(data_set_eval["test"][0]))
        data_set_eval["test"][0][:] = data_set_eval["test"][0][random_indices]
        data_set_eval["test"][1][:] = data_set_eval["test"][1][random_indices]

    main_header_name = "test_arrays"
    model_dependent_test_header_name = "test_array_" + unique_model_name

    # Start with modifiying the test_arrays.h file
    header_file_header = f"#ifndef INC_TEST_ARRAYS_H_\n#define INC_TEST_ARRAYS_H_\n"
    header_to_be_imported = f'#include "' + model_dependent_test_header_name + f'.h"\n'
    header_file_footer = f"#endif /* INC_TEST_ARRAYS_H_ */\n"
    header_file = (
        header_file_header
        + f"\n\n"
        + header_to_be_imported
        + f"\n\n"
        + header_file_footer
    )

    f = open(header_path + main_header_name + ".h", "w")
    f.write(header_file)
    f.close()

    # now create a new model dependent file
    c_data_x_test_eval, c_data_y_test_eval = data_set_eval["test"]
    header_file_header = (
        f"#ifndef INC_TEST_ARRAYS_VALUES_H_\n#define INC_TEST_ARRAYS_VALUES_H_\n"
    )
    header_file_footer = f"#endif /* INC_TEST_ARRAYS_VALUES_H_ */\n"

    # Convert Python array to C initialization
    c_test_input = python_matrix_to_c_matrix(c_data_x_test_eval, "test_input")
    c_test_output = python_matrix_to_c_matrix(c_data_y_test_eval, "test_output")

    scaler_string = ""

    scaler_string += (
        f"float polynomial_degree = \t"
        + f"".join(map(str, [polynomial_degree]))
        + f";\n"
    )

    if use_arrhenius_correction:
        scaler_string += f"bool use_arrhenius_correction = 1;\n"
    else:
        scaler_string += f"bool use_arrhenius_correction = 0;\n"

    if use_arrhenius_correction_with_factor:
        scaler_string += f"bool use_arrhenius_correction_with_factor = 1;\n"
    else:
        scaler_string += f"bool use_arrhenius_correction_with_factor = 0;\n"

    if use_min_max_scaler:
        scaler_string += f"bool use_min_max_scaler = 1;\n"
    else:
        scaler_string += f"bool use_min_max_scaler = 0;\n"

    if use_standard_scaler:
        scaler_string += f"bool use_standard_scaler = 1;\n"
    else:
        scaler_string += f"bool use_standard_scaler = 0;\n"

    if scale_y_data:
        scaler_string += f"bool scale_y_data = 1;\n"
    else:
        scaler_string += f"bool scale_y_data = 0;\n"

    scaler_string += f"\n"

    scaler_string += (
        f"float arrhenius_b = \t" + f"".join(map(str, [arrhenius_b])) + f";\n"
    )
    scaler_string += (
        f"float arrhenius_c = \t" + f"".join(map(str, [arrhenius_c])) + f";\n"
    )
    scaler_string += (
        f"float min_max_scaler_x_min = \t" + f"".join(map(str, [x_min])) + f";\n"
    )
    scaler_string += (
        f"float min_max_scaler_x_max = \t" + f"".join(map(str, [x_max])) + f";\n"
    )
    scaler_string += (
        f"float min_max_scaler_y_min = \t" + f"".join(map(str, [y_min])) + f";\n"
    )
    scaler_string += (
        f"float min_max_scaler_y_max = \t" + f"".join(map(str, [y_max])) + f";\n"
    )
    scaler_string += (
        f"float standard_scaler_x_mean = \t" + f"".join(map(str, [x_mean])) + f";\n"
    )
    scaler_string += (
        f"float standard_scaler_x_std = \t" + f"".join(map(str, [x_std])) + f";\n"
    )
    scaler_string += (
        f"float standard_scaler_y_mean = \t" + f"".join(map(str, [y_mean])) + f";\n"
    )
    scaler_string += (
        f"float standard_scaler_y_std = \t" + f"".join(map(str, [y_std])) + f";\n"
    )

    header_file = (
        header_file_header
        + f"\n\n"
        + scaler_string
        + f"\n\n"
        + c_test_input
        + f"\n\n"
        + c_test_output
        + f"\n\n"
        + header_file_footer
    )

    f = open(header_path + model_dependent_test_header_name + ".h", "w")
    f.write(header_file)
    f.close()


def quantize_data(data, discretize_minimum, discretize_delta):
    """
    Quantizes the input data based on the specified minimum and delta values.

    Parameters:
        data (numpy.ndarray): The input data to be quantized.
        discretize_minimum (float): The minimum value for discretization.
        discretize_delta (float): The step size for discretization.

    Returns:
        numpy.ndarray: The quantized data.
    """

    data = data - discretize_minimum
    data = data / discretize_delta
    data = np.round(data, 0)
    data = data * discretize_delta
    data = data + discretize_minimum
    return data


def df_2_arrays(df, label, feature_keys):
    """
    Convert a dataframe to numpy arrays.

    Parameters:
        df: the dataframe to convert
        label: the label column to use
        feature_keys: the feature keys to use

    Returns:
        x_arr: the feature array
        y_arr: the label array
    """

    x_arr = df[feature_keys].to_numpy()
    y_arr = df[label].to_numpy()
    return x_arr, y_arr


def mix_split_df(df, validation_split, random_state):
    """
    Split the dataframe into training and validation dataframes.

    Parameters:
        df: the dataframe to split
        validation_split: the percentage of the dataframe to be used for validation
        random_state: the random state to use for the split

    Returns:
        df_train: the training dataframe
        df_validation: the validation dataframe
    """

    # Shuffle the data set before splitting the dataset
    df = df.sample(frac=1, random_state=random_state)

    # Split dataframe into validation and train set
    split_index = int(validation_split * len(df))
    df_validation = df.iloc[:split_index]
    df_train = df.iloc[split_index:]
    return df_train, df_validation


def get_set(
    df,
    labels,
    feature_keys=[],
    validation_split=0.2,
    test_split=0.1,
    output_intervals_for_test=None,
    label_for_test_intervals=None,
    random_state=1,
    label_name="",
):
    """
    Get the training, validation, and test data from a dataframe.

    Parameters:
        df: pandas DataFrame
            The dataset containing features and labels.
        labels: str or list of str
            The label column(s) to use for outputs. Can be a single string or a list of strings for multiple outputs.
        feature_keys: list of str
            The feature columns to use.
        validation_split: float
            The proportion of the dataset to include in the validation split (between 0 and 1).
        test_split: float
            The proportion of the dataset to include in the test split (between 0 and 1).
        output_intervals_for_test: dict or list, optional
            Intervals of values to use for the test data. If `label_for_test_intervals` is provided, this applies to that label.
            If `label_for_test_intervals` is not provided, and `labels` is a list, this should be a dict where keys are labels and values are lists of intervals for that label.
            If `labels` is a single label or you want to apply the same intervals to all labels, this can be a list of intervals.
        label_for_test_intervals: str or list of str, optional
            The label(s) to use for selecting the test set based on intervals. If not provided, defaults to `labels`.
        random_state: int
            The random seed for reproducibility.
        label_name: str
            An optional name for the label.

    Returns:
        data: dict
            A dictionary containing the training, validation, and test data.
    """

    if isinstance(labels, str):
        labels = [labels]
    if label_for_test_intervals is not None:
        if isinstance(label_for_test_intervals, str):
            test_labels = [label_for_test_intervals]
        else:
            test_labels = label_for_test_intervals
    else:
        test_labels = labels

    df_test = pd.DataFrame()
    df_train_validation = df.copy()

    if output_intervals_for_test:
        if isinstance(output_intervals_for_test, dict):
            for label in test_labels:
                if label in output_intervals_for_test:
                    intervals = output_intervals_for_test[label]
                    for interval in intervals:
                        condition = (df_train_validation[label] >= interval[0]) & (
                            df_train_validation[label] <= interval[1]
                        )
                        df_test = pd.concat([df_test, df_train_validation[condition]])
                        df_train_validation = df_train_validation.drop(
                            df_train_validation[condition].index
                        )
        else:
            for interval in output_intervals_for_test:
                condition = np.ones(len(df_train_validation), dtype=bool)
                for label in test_labels:
                    label_condition = (df_train_validation[label] >= interval[0]) & (
                        df_train_validation[label] <= interval[1]
                    )
                    condition = condition & label_condition
                df_test = pd.concat([df_test, df_train_validation[condition]])
                df_train_validation = df_train_validation.drop(
                    df_train_validation[condition].index
                )

        df_test = df_test.drop_duplicates()
        df_train, df_validation = mix_split_df(
            df_train_validation, validation_split, random_state
        )

    else:
        total_split = validation_split + test_split
        if total_split >= 1.0:
            raise ValueError(
                "The sum of validation_split and test_split must be less than 1.0"
            )

        df_train_validation, df_test = train_test_split(
            df, test_size=test_split, random_state=random_state, shuffle=True
        )

        validation_ratio = validation_split / (1.0 - test_split)
        df_train, df_validation = train_test_split(
            df_train_validation,
            test_size=validation_ratio,
            random_state=random_state,
            shuffle=True,
        )

    x_train, y_train = df_2_arrays(df_train, labels, feature_keys)
    x_validation, y_validation = df_2_arrays(df_validation, labels, feature_keys)
    x_test, y_test = df_2_arrays(df_test, labels, feature_keys)

    data = {
        "train": (x_train, y_train),
        "validation": (x_validation, y_validation),
        "test": (x_test, y_test),
        "df_train": df_train,
        "df_validation": df_validation,
        "df_test": df_test,
        "label": labels,
        "label_name": label_name,
    }
    return data


def get_set_nn(
    df,
    label,
    feature_keys=[],
    validation_split=0.2,
    output_intervals_for_test=[[1, 2]],
    random_state=1,
    oversample=False,
    oversample_bins=10,
):
    """
    Get the training and test data from a dataframe.

    Parameters:
        df: the dataframe
        label: either string or list of strings indicating the label columns to use
        feature_keys: the feature keys to use
        validation_split: the percentage of the dataframe to be used for testing
        output_intervals_for_test: the output intervals to use for the test data
        random_state: the random state to use for the split
        test_interval_size: nr of measurements in every test interval (if available) - if None is passed, all measurements are taken
        oversample: if set to true the training and validation data are oversampled (underrepresneted values are repeated)
        oversample_bins: bins of the histogramm that is taken as the base for oversamplingS
    Returns:
        data: a dictionary containing the training, validation and test data
    """

    # Extract the test data
    df_test = df.head(0).copy()
    df_train_validation = df.copy()

    # convert to short list if
    if type(label) is str:
        label = [label]

    # convert to 3d list if list is 2d
    if not isinstance(output_intervals_for_test[0][0], list):
        output_intervals_for_test = [output_intervals_for_test]

    for label_index, single_label in enumerate(label):
        # go through intervals
        for interval_index in range(len(output_intervals_for_test[label_index])):
            # select df with label within the interval
            df_interval = df_train_validation.loc[
                (
                    df_train_validation[single_label]
                    >= output_intervals_for_test[label_index][interval_index][0]
                )
                & (
                    df_train_validation[single_label]
                    <= output_intervals_for_test[label_index][interval_index][1]
                )
            ]

            # add sampled values to df_test
            df_test = pd.concat([df_test, df_interval])
            # drop sampels from training data
            df_train_validation.drop(df_interval.index, inplace=True)

    # Split the dataframe
    df_train, df_validation = mix_split_df(
        df_train_validation, validation_split, random_state
    )

    if oversample:
        for single_label in label:
            df_train = oversample_df(df_train, bins=oversample_bins, label=single_label)
            df_validation = oversample_df(
                df_validation, bins=oversample_bins, label=single_label
            )

    # Convert dataframe to arrays
    x_train, y_train = df_2_arrays(df_train, label, feature_keys)
    x_validation, y_validation = df_2_arrays(df_validation, label, feature_keys)
    x_test, y_test = df_2_arrays(df_test, label, feature_keys)

    # Return a dictionary of the whole dataset
    data = {
        "train": (x_train, y_train),
        "validation": (x_validation, y_validation),
        "test": (x_test, y_test),
        "df_train": df_train,
        "df_validation": df_validation,
        "df_test": df_test,
        "label": label,
        "feature_keys": feature_keys,
    }
    return data


def sort_out_NaN_features(df, feature_keys):
    """
    Every datapoint is sorted out, if one of its features is equal to NaN

    Parameters:
        df: the dataframe containing the features
        feature_keys: strings that are equal to the column names of the features in the dataframe
    Returns:
        df_new: a dataframe without the rows that contain features equal to NaN
    """
    df_new = df.copy()
    for feature_key in feature_keys:  # go through all the column names
        # count the nr of NaN entries in the column
        nr_NaN = df_new[feature_key].isnull().sum()
        if nr_NaN > 0:
            # delete all those entries
            print(feature_key, "= NaN in", nr_NaN, "rows")

            # delete those rows
            df_new = df_new.loc[df_new[feature_key].notnull()]
            print(nr_NaN, "rows deleted")

    return df_new


def oversample_df(df, bins=20, label="Temperature", random_state=1):
    """
    Oversample the dataframe by creating a new dataframe with the same data as the original, but with a new label.

    Parameters:
        df: the dataframe to oversample
        bins: the number of bins to create
        label: the label to oversample
        random_state: the random state to use
    Returns:
        df_new: the oversampled dataframe
    """

    # first we define the slices
    slices = np.linspace(df[label].min(), df[label].max(), num=bins + 1)
    # Determine the biggest slice
    biggest_slice = 0
    for i, sl in enumerate(slices[1:]):
        # length of the current slice
        sl_len = len(df.loc[(df[label] > slices[i]) & (df[label] < sl)])
        # print(biggest_slice)
        if sl_len > biggest_slice:
            biggest_slice = sl_len

    df_new = pd.DataFrame()
    # Go through the slices and sample up
    for i, sl in enumerate(slices[1:]):
        # Select the dataframe of the slice
        df_slice = df.loc[(df[label] > slices[i]) & (df[label] < sl)]
        # upsample slice
        if len(df_slice) > 0:  # only if there is something to sample up
            df_slice = df_slice.sample(
                n=biggest_slice, replace=True, random_state=random_state
            )
            # append to new df
            df_new = pd.concat([df_new, df_slice])

    return df_new


def list_2_json(list_, filename, folder="data/feature_selections/"):
    """
    Write a list to a json file.

    Parameters:
        list_: the list to write to a json file
        filename: the name of the file to write to
        folder: the folder to write to

    Returns:
        None
    """

    filepath = os.path.join(folder, filename)
    with open(filepath, "w") as f:
        f.write(json.dumps(list_))


def json_2_list(filename, folder="data/feature_selections/"):
    """
    Given a filename, open the file and return the contents as a list.

    Parameters:
        filename: the name of the json file
        folder: the folder where the file is located (default: "data/feature_selections/")

    Returns:
        list_: the contents of the json file as a list
    """
    filepath = os.path.join(folder, filename)
    with open(filepath, "r") as f:
        return json.load(f)
