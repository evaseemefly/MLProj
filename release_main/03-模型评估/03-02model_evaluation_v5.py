import csv
from typing import Optional, Any
import random
import arrow
import pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
# from keras.src.layers import LSTM, Dropout, Dense
from pandas import DatetimeIndex
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# 可视化结果（如果需要）
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
import xarray as xr
import codecs
import datetime
import math
import joblib
# 先从海浪数据中提取出经纬度，时间，风，海浪高度
# 解析单个文件，并存于字典内
from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys


def load_customer_model(model_path: str) -> Any:
    """
        加载模型
    :param model_path:
    :return:
    """
    if pathlib.Path(model_path).exists():
        loaded_model = load_model(model_path)
        """加载后的模型"""
        print(loaded_model.summary())
        return loaded_model
    return None


def model_predict(model_path: str, vals: pd.Series) -> pd.Series:
    """
        加载指定模型，并将原始预报数据——vals 进行订正并返回
    :param model_path:
    :param vals:
    :return:
    """
    X = np.nan_to_num(vals, nan=0.0)
    # 注意由于 model.add(Dense(25)) 加入了全连接层，最后一步对每个时间输出25维结果，所以暂时取出第一个维度的数据
    model = load_customer_model(model_path)
    X_pred = model.predict(X)
    return X_pred


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> pd.Series:
    """
        计算两个形状相同的 NumPy 数组的行均方根误差 (RMSE)，并以 pd.Series 形式返回。
    :param x_series:
    :param y_series:
    :return:
    """
    assert y_true.shape == y_pred.shape, "输入数组的形状必须相同"
    # 计算差值的平方
    squared_errors = (y_true - y_pred) ** 2

    # 计算每列的均方误差 (MSE)
    # mse_per_row = squared_errors.mean(axis=1)  # axis=0 表示按列计算均值

    # 使用 np.nanmean 代替 np.mean
    # 它会在计算均值时忽略掉 NaN 值
    mse_per_row = np.nanmean(squared_errors, axis=1)

    # 计算每列的均方根误差 (RMSE)
    rmse_per_row = np.sqrt(mse_per_row)
    return rmse_per_row


def main_evaluation(model_path: str, forecast_data_path: str, real_data_path: str, scaler_forecast_path: str,
                    scaler_realdata_path: str, out_put_rmse_path: str, dataset_count=61, out_put_pic_path: str = None,
                    warmup_count: int = 0):
    # step1: 读取 预报及实况数据集
    df_forecast = pd.read_csv(forecast_data_path, encoding='utf-8', index_col=0)
    df_realdata = pd.read_csv(real_data_path, encoding='utf-8', index_col=0)
    # TODO:[*] 25-07-13 注意V5版开始模型加入了预热数据，所以需要剔除预热数据的评估
    # 将 [:warmup_count+1] 的数据剔除掉，只评估[warmup_count:]的数据
    start_index = 0
    df_forecast_standardized = df_forecast.iloc[start_index:dataset_count, :]
    df_realdata_standardized = df_realdata.iloc[start_index:dataset_count, :]

    # step2: 生成验证数据集
    split_count = math.ceil(df_forecast.shape[1] * 0.2)
    split_df_forecast = df_forecast_standardized[df_forecast_standardized.columns[-split_count:]]
    split_df_realdata = df_realdata_standardized[df_realdata_standardized.columns[-split_count:]]

    rows: int = split_df_forecast.shape[0]
    cols: int = split_df_forecast.shape[1]
    # TODO:[-] 25-05-28 注意原始数据中: forecast (72,732), real (72,733)
    X = split_df_forecast.values.T.reshape(cols, rows, 1)
    # TODO:[*] 25-05-11 注意 y 中有存在 nan
    y = split_df_realdata.values.T.reshape(cols, rows, 1)
    # step3-2:对数据进行归一化
    # 拍扁数据为二维数组（n*timesteps, feature）进行归一化
    X_flat = X.reshape(-1, 1)
    y_flat = y.reshape(-1, 1)

    # step4: 加载归一化器
    scaler_forecast = joblib.load(scaler_forecast_path)
    scaler_realdata = joblib.load(scaler_realdata_path)

    # step5:
    # 分别为 X 和 y 定义归一化器（当然如果两者量纲一致，可用同一个 scaler）
    X_scaled = scaler_forecast.transform(X_flat)
    """归一化后的——预报训练集"""
    y_scaled = scaler_realdata.transform(y_flat)
    """归一化后的——实况训练集"""
    # 将归一化后的二维数据恢复为原来的3D形状
    X = X_scaled.reshape(X.shape)
    y = y_scaled.reshape(y.shape)
    # TODO:[-] 25-06-16 加入对于0值的过滤
    # X = np.nan_to_num(X, nan=0.0)
    # TODO:[*] 25-06-16 此处可能存在错误
    # TODO:[-] 25-07-13 ERROR: ValueError: Input 0 of layer "sequential" is incompatible with the layer: expected shape=(None, 64, 1), found shape=(None, 60, 1)
    # 由于剔除了预热数据导致的错误
    X_pred = model_predict(model_path, X)

    # TODO:[-] 25-06-16 此处缺少反归一化，导致误差出错
    # random_num = X_pred.shape[2]
    # random_index = random.randint(0, random_num)
    X_fit_random = X_pred[:, :, 0]
    X_denormalize = scaler_forecast.inverse_transform(X_fit_random)
    # TODO:[*] 25-06-17 为何需要转置
    X_denormalize = X_denormalize.T

    # TODO;[-] 25-07-13 剔除预热数据在使用模型修订后进行剔除预热数据的操作。预测数据以及对比的实况数据均需要剔除预热数据

    # step7: 计算 RMSE
    # TODO:[*] 25-07-13 X_denormalize : ndarray ; split_df_realdata: DataFrame
    ndarray_realdata = split_df_realdata.to_numpy()
    eliminate_realdata_ndarray: np.ndarray = ndarray_realdata[warmup_count:, :]
    eliminate_forecast_ndarray: np.ndarray = X_denormalize[warmup_count:, :]

    rmse_series = compute_rmse(eliminate_forecast_ndarray, eliminate_realdata_ndarray)
    # step8: 存储 RMSE
    now_arrow: arrow.Arrow = arrow.now()
    date_str: str = now_arrow.date().isoformat()
    rmse_file_name: str = f'rmse_forecast_{date_str}.csv'
    rmse_full_path: str = pathlib.Path(out_put_rmse_path) / rmse_file_name
    rmse_df = pd.DataFrame(rmse_series, columns=['RMSE'])
    rmse_df.to_csv(rmse_full_path)
    print(f'[-] 输出的RMSE存储路径为:{rmse_full_path}')

    pass


def main():
    model_path: str = r'Z:\02TRAINNING_MODEL\fit_model_v6_250713.h5'
    forecast_path: str = r'Z:\01TRAINNING_DATA\250713\2024_warmup_dataset_forecast_aligned_250713.csv'
    realdata_path: str = r'Z:\01TRAINNING_DATA\250713\2024_warmup_dataset_realdata.csv'
    forecast_scaler_path: str = r'Z:\01TRAINNING_DATA\scaler\250713\scaler_forecast.sav'
    real_scaler_path: str = r'Z:\01TRAINNING_DATA\scaler\250713\scaler_realdata.sav'
    out_put_pic_path: str = r'Z:\04TRAINNING_EVALUATION_PIC\MODEL_V5'
    out_put_rmse_path: str = r'Z:\04TRAINNING_EVALUATION_PIC\MODEL_V5'

    warmup_count = 4

    # TODO:[-] 25-07-13 此处加入了数据预热 warmup_count =4, 会忽略 [0:warmup_count)
    main_evaluation(model_path, forecast_path, realdata_path, forecast_scaler_path, real_scaler_path, out_put_rmse_path,
                    64,
                    out_put_pic_path, warmup_count=4)
    pass


if __name__ == '__main__':
    main()
