"""
Module for data analyze.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import Any


def output_csv(df: pd.DataFrame, output_data_path: str):
    """
    Output df as a csv file to output_data_path.
    Args:
        df:Data
        output_data_path:Output path of the csv file
    """
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    df.to_csv(output_data_path)


def make_missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make missing values table of df.
    Args:
        df:Data
    Returns:
        Number and percentage of missing values for each column.
    """
    null_val = df.isnull().sum()
    missing_rate = df.isnull().sum() / len(df)
    missing_table = pd.concat([null_val, missing_rate], axis=1)
    missing_table = missing_table.rename(
        columns={0: 'missing_num', 1: 'missing_rate'})
    missing_table = missing_table[missing_table['missing_num'] != 0]
    return missing_table


def top_correlation_column(
        df: pd.DataFrame,
        target: str,
        num: int = None) -> Any:
    """
    Extract the columns with high correlation to the target in df.
    Args:
        df:Data
        target:Column of target
        num:Number of columns to be extracted
            If not set, it is the number of columns of df.corr().
    Returns:
        Column Index, following format
        Index(['column1', 'column2'], dtype='object')
    """
    corrmat = df.corr()
    if num is None:
        num = len(corrmat)
    cols = corrmat.abs().nlargest(num, target)[target].index
    return cols


def plot_corr_heatmap(df, target, num=None):
    """
    Show heatmap the columns with high correlation to the target in df.
     Args:
        df:Data
        target:Column of target
        num:Number of columns to be extracted
            If not set, it is the number of columns of df.corr().
    """
    cols = top_correlation_column(df, target, num)
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
