import csv
from typing import Optional, Any
import random
import arrow
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
import pathlib
import xml.etree.ElementTree as ET
import xarray as xr
import codecs
import datetime

# 先从海浪数据中提取出经纬度，时间，风，海浪高度
# 解析单个文件，并存于字典内
from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sys


def main():
    rmse_source_path: str = r'Z:\03TRAINNING_EVALUATION_DATA\rmse_forecast_source_250609.csv'
    rmse_model_path: str = r'Z:\04TRAINNING_EVALUATION_PIC\MODEL_V5\rmse_forecast_2025-07-13.csv'
    rmse_source = pd.read_csv(rmse_source_path)
    rmse_model_fit = pd.read_csv(rmse_model_path)
    out_put_pic_dic: str = r'Z:\04TRAINNING_EVALUATION_PIC\MODEL_V5'
    file_name_model_fit: str = pathlib.Path(rmse_model_path).name + '.png'
    out_put_pic_fullpath: str = pathlib.Path(out_put_pic_dic) / file_name_model_fit

    series_model_fit = pd.Series(rmse_model_fit['RMSE'])
    series_SOURCE = pd.Series(rmse_source['RMSE'])

    # 创建图表
    plt.figure(figsize=(8, 5))  # 设置图像大小
    plt.plot(series_model_fit, label="fit model forecast", marker='o')  # 绘制第一条线
    plt.plot(series_SOURCE, label="source model forecast", marker='s')  # 绘制第二条线

    # 添加标题和标签
    plt.title("Series Comparison")
    plt.xlabel("Index")
    plt.ylabel("Values")
    plt.legend()  # 显示图例
    plt.grid()  # 添加网格线
    plt.savefig(out_put_pic_fullpath)
    # 显示图表
    plt.show()
    pass


if __name__ == '__main__':
    main()
