import arrow
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
# TODO:[-] 25-06-03 keras.src 路径是 TensorFlow 2.11 及更高版本中集成在 TensorFlow 内部的 Keras 3 中使用的。
# from keras.src.layers import LSTM, Dropout, Bidirectional, Dense, Masking
from keras.layers import LSTM, Dropout, Bidirectional, Dense, Masking
from pandas import DatetimeIndex
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from tensorflow.keras.losses import MeanSquaredError
# 可视化结果（如果需要）
import matplotlib.pyplot as plt
import os
import pathlib
import xml.etree.ElementTree as ET
import xarray as xr
import codecs
import datetime

# 先从海浪数据中提取出经纬度，时间，风，海浪高度
# 解析单个文件，并存于字典内
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# from utils import rmse

def realdata2sort(read_path: str, out_path: str):
    df = pd.read_csv(read_path, encoding='utf-8', index_col=0)
    df_sorted = df.sort_index(axis=1, ascending=True)
    diff_cols = df.columns.difference(df_sorted.columns)
    print(f'dataframe columns 顺序有误:{diff_cols}')
    if len(diff_cols) > 0:
        df_sorted.to_csv(out_path, encoding='utf-8')


def main():
    # TODO:[-] 25-06-08 新加入的razer配置
    # TODO:[-] 25-07-13 对于 预报 | 实况 数据加入了 warmup 预热数据集
    forecast_path: str = r'Z:\01TRAINNING_DATA\250713\2024_warmup_converted_dataset_250713.csv'
    forecast_aligned_path: str = r'Z:\01TRAINNING_DATA\250713\2024_warmup_dataset_forecast_aligned_250713.csv'
    realdata_path: str = r'Z:\01TRAINNING_DATA\250713\2024_warmup_dataset_realdata.csv'
    '''按照三小时一个提取的实况数据路径'''

    # step1: 加载标准化后的 预报 | 实况 数据集
    # shape: (61,732)
    df_forecast = pd.read_csv(forecast_path, encoding='utf-8', index_col=0)
    # shape:(61,731)
    df_realdata = pd.read_csv(realdata_path, encoding='utf-8', index_col=0)
    """
        预报场缺失的时间
        ['2024-03-13 00:00:00', '2024-04-01 00:00:00', '2024-04-11 12:00:00',
       '2024-04-12 00:00:00', '2024-04-13 12:00:00', '2024-04-15 00:00:00',
       '2024-04-18 00:00:00', '2024-04-19 00:00:00', '2024-04-20 12:00:00',
       '2024-05-08 00:00:00', '2024-05-18 00:00:00', '2024-05-18 12:00:00',
       '2024-05-19 00:00:00', '2024-05-19 12:00:00', '2024-05-20 00:00:00',
       '2024-05-20 12:00:00', '2024-05-27 00:00:00', '2024-06-14 00:00:00',
       '2024-07-10 12:00:00', '2024-07-11 00:00:00', '2024-07-18 00:00:00',
       '2024-07-22 00:00:00', '2024-07-22 12:00:00', '2024-08-09 00:00:00',
       '2024-08-09 12:00:00', '2024-08-10 00:00:00', '2024-08-10 12:00:00',
       '2024-08-18 00:00:00', '2024-09-19 00:00:00', '2024-12-13 12:00:00',
       '2024-12-14 12:00:00', '2024-12-15 12:00:00']
    """
    # 获取缺失的数据列
    missing_cols = df_realdata.columns.difference(df_forecast.columns)
    print(missing_cols)

    # 以 实况 df为基准，将 预报 df 向 实况对齐
    df_forecast_aligned = df_forecast.reindex(columns=df_realdata.columns)
    print(f'预报 原始数据:shape{df_forecast.shape}|对齐后数据:shape{df_forecast_aligned.shape}')
    """
        axis=1: 这个参数告诉 Pandas 我们要排序的是列索引（columns），而不是行索引（axis=0，默认值）。
        ascending=True: 这是默认行为，表示升序排列。对于 datetime 类型的列，就是从最早的时间排到最晚的时间。
    """
    df_forecast_aligned = df_forecast_aligned.sort_index(axis=1, ascending=True)
    df_forecast_aligned.to_csv(forecast_aligned_path)
    pass


if __name__ == '__main__':
    realdata_path: str = r'Z:\01TRAINNING_DATA\250713\2024_warmup_dataset_realdata.csv'
    realdata_path_aligned: str = r'Z:\01TRAINNING_DATA\250713\2024_warmup_dataset_realdata_aligned_250713.csv'
    main()
    # realdata2sort(realdata_path, realdata_path_aligned)
    pass
