# -*- coding: utf-8 -*-

import json
import os
import numpy as np
import pandas as pd


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
    label,
    feature_keys=[],
    validation_split=0.2,
    output_intervals_for_test=[[1, 2]],
    random_state=1,
):
    """
    Get the training and test data from a dataframe.

    Parameters:
        df: the dataframe
        label: the label column to use
        feature_keys: the feature keys to use
        validation_split: the percentage of the dataframe to be used for testing
        output_intervals_for_test: the output intervals to use for the test data
        random_state: the random state to use for the split

    Returns:
        data: a dictionary containing the training, validation and test data
    """

    # Extract the test data
    df_test = df.head(0).copy()
    df_train_validation = df.copy()
    for index in range(np.shape(output_intervals_for_test)[0]):
        df_test = pd.concat(
            [
                df_test,
                df.loc[
                    (df[label] >= output_intervals_for_test[index][0])
                    & (df[label] <= output_intervals_for_test[index][1])
                ],
            ]
        )
        df_train_validation.drop(
            df_train_validation[
                (df_train_validation[label] >= output_intervals_for_test[index][0])
                & (df_train_validation[label] <= output_intervals_for_test[index][1])
            ].index,
            inplace=True,
        )

    # Split the dataframe
    df_train, df_validation = mix_split_df(
        df_train_validation, validation_split, random_state
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
    }
    return data


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
