'''
Module for data analyze.
'''

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import math


def output_csv(df: pd.DataFrame, output_data_path):
    """
    Output df to csv(output_data_path)
    Args:
        df:
        output_data_path:
    """
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    df.to_csv(output_data_path)


def make_missing_table(df) -> pd.DataFrame:
    """
    Make missing_table of df
    Args:
        df:

    Returns:

    """
    null_val = df.isnull().sum()
    missing_rate = df.isnull().sum() / len(df)
    missing_table = pd.concat([null_val, missing_rate], axis=1)
    missing_table = missing_table.rename(
        columns={0: 'missing_num', 1: 'missing_rate'})
    missing_table = missing_table[missing_table['missing_num'] != 0]
    return missing_table


def move_column_end(df, column):
    '''
    Move column to last
    :param df:
    :param column:
    :return:
    '''
    columns = df.columns.tolist()
    columns.remove(column)
    columns.append(column)
    df = df[columns]
    return df


def transform_yeo_johnson():
    return


def plot_relation_target(df, target_column, output_data_path="relation.png"):
    '''
    Plot relation to the target_column
    '''
    # relation to the target
    col_n = df.shape[1]
    y_line = math.ceil(col_n / 5)
    fig = plt.figure(figsize=(16, 3 * y_line))
    for i in np.arange(col_n):
        ax = fig.add_subplot(y_line, 5, i + 1)
        sns.regplot(x=df.iloc[:, i], y=df[target_column])
    plt.tight_layout()
    # plt.show()
    plt.savefig(output_data_path)


def top_correlation_column(df, target_column, num=None):
    """
    :param df:pandas data frame
        data
    :param target_column:str
    :param num:int
        If not set, it is the number of columns of df.corr().
    """
    corrmat = df.corr()
    if num is None:
        num = len(corrmat)
    cols = corrmat.abs().nlargest(num, target_column)[target_column].index
    return cols


def plot_corr_heatmap(df, target_column, num=None):
    """
    Visualize heatmap top num target_columns abs value of correlation with collumn
    :param df:pandas data frame
        data
    :param target_column:str
    :param num:int
        If not set, it is the number of columns of df.corr().
    """

    cols = top_correlation_column(df, target_column, num)

    if num is None:
        num = len(cols)

    cm = df[cols].corr()
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(num, num))
    sns.heatmap(
        cm,
        cbar=True,
        annot=True,
        square=True,
        ax=ax,
        fmt='.2f',
        annot_kws={
            'size': 15},
        yticklabels=cols.values,
        xticklabels=cols.values)
    ax.set_ylim(num, 0)
    plt.show()


def top_num_index(num, l):
    '''
    Top num index list of l
    '''
    li = np.array(l).argsort()[::-1]
    tni = li[:num]
    return tni


def make_feature_importances_table(x_train, model, num=10):
    '''
    Top num columns feature importance using learned model.
    :return pd.DataFrame
    '''
    feature_importances_index = top_num_index(num, model.feature_importances_)
    feature_importances_table = pd.DataFrame(
        {"column": x_train.columns.values[feature_importances_index],
         "importances": model.feature_importances_[feature_importances_index]})
    return feature_importances_table


def plot_feature_importances(x_train, model, num=10):
    '''
    Plot top num columns feature importance using learned model.
    '''
    feature_importances_table = make_feature_importances_table(
        x_train, model, num=num)
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.set_xlabel("feature importance")
    plt.tight_layout()
    sns.barplot(x=feature_importances_table["importances"],
                y=feature_importances_table["column"], orient='h')
