import csv
from typing import Optional, List

import arrow
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from pandas import DatetimeIndex
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import pathlib
import xml.etree.ElementTree as ET
import xarray as xr
import codecs
import datetime


def realdata2warmup(read_path: str, out_put_path: str, warm_up_count: int):
    """
        提取实况数据并掺入预热数据
    :param read_path:
    :param out_put_path:
    :param warm_up_count:
    :return:
    """
    df_source = pd.read_csv(read_path, index_col=0)
    # 从index=1 columns 开始，将前一个column 的列数据取出，提取头warm_up_count个数据进行填充
    columns = df_source.columns
    start_index = 1
    dict_merge = {}
    for previous_col, current_col in zip(columns[:-1], columns[1:]):
        print(previous_col, current_col)
        previous_series = df_source[previous_col]
        """前一个series"""
        current_series = df_source[current_col]
        """当前series"""
        # 从 前一个series 中取出 [:3]
        previous_vals = previous_series.iloc[0:0 + warm_up_count]
        current_merge_series = pd.concat([previous_vals, current_series])
        dict_merge[current_col] = current_merge_series
    df_merge: pd.DataFrame = pd.DataFrame(dict_merge)
    return df_merge


def main():
    """
        step1: 读取处理后的时间间隔为3hour的fub实况数据
        step2: 对实况数据按照 warm_up_count 进行拼接
    :return:
    """

    FUB_READ_PATH: str = r'Z:\01TRAINNING_DATA\FUB\MF01001\2024_local_df_utc_183_split.csv'
    OUT_PUT_FUB_PATH: str = r'Z:\SOURCE_MERGE_DATA\REALDATA\2024_fub_realdata_warmup_dataset.csv'
    df_merge = realdata2warmup(FUB_READ_PATH, OUT_PUT_FUB_PATH, 4)
    df_merge.to_csv(OUT_PUT_FUB_PATH, index=False)
    pass


if __name__ == '__main__':
    main()
