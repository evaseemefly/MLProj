from typing import List

import arrow
import joblib
import numpy as np
import pandas as pd
import pytz
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

from tornado.gen import Return

from utils.common import get_files
from utils.sorts import get_wind_sort_key


def read_stations(read_path: str) -> List[dict]:
    """
        获取站点字典数组
    :param read_path:
    :return:
    """
    df = pd.read_csv(read_path)
    res: List[dict] = []
    code_col = 'code'
    lat_col = 'lat'
    lng_col = 'lng'
    for index, row in df.iterrows():
        code = row[code_col]
        lat = row[lat_col]
        lng = row[lng_col]
        temp_dict = {'code': code, 'lat': lat, 'lng': lng}
        res.append(temp_dict)
    return res


def generate_time_range(start: arrow.Arrow, end: arrow.Arrow, step_hour: int = 12) -> List[arrow.Arrow]:
    """
        根据起止时间生成对应，并按照时间步长 step_hour 创建 时间集合
    :param start:
    :param end:
    :param step_hour:
    :return:
    """
    # 2. 初始化一个空列表用于存放结果
    result_arrows = []

    # 3. 初始化当前时间为起始时间
    current_arrow = start

    # 4. 循环直到当前时间超过结束时间
    while current_arrow <= end:
        # 将当前时间添加到结果列表中
        result_arrows.append(current_arrow)
        # 将当前时间增加12小时，作为下一次循环的时间点
        current_arrow = current_arrow.shift(hours=step_hour)

    return result_arrows


def read_ds_values(root_path: str, dt_arrow: arrow.Arrow, code: str, lat: float, lng: float,
                   time_count: int = 72) -> dict:
    """
        根据当前预报时间 dt_arrow 获取对应的 文件名: GRAPES_2024010100_240h_UV.nc
        读取指定文件并根据 lat,lng 获取指定 time_count 长度的预报数据
        eg:
            code_2024.csv =>
                            [yyyymmddhhmmss_u,yyyymmddhhmmss_v,...]
    :param dt_arrow:
    :param code:
    :param lat:
    :param lng:
    :param time_count:
    :return:
    """
    dt_str: str = dt_arrow.format('YYYYMMDDHH')
    target_file_name: str = f'GRAPES_{dt_str}_240h_UV.nc'
    df_dict: dict = {}
    # 1: 判断指定文件是否存在
    if pathlib.Path(root_path).is_dir():
        target_path = pathlib.Path(root_path) / target_file_name
        if target_path.exists():
            ds: xr.Dataset = xr.open_dataset(str(target_path))
            if ds is not None:
                # 从该文件中提取指定经纬度的时序数据
                filter_ds = ds.sel(latitude=lat, longitude=lng, method='nearest')
                # 分别取出 u 与 v 分量
                u_vals = filter_ds['UGRD_10maboveground'].values[:time_count]
                v_vals = filter_ds['VGRD_10maboveground'].values[:time_count]
                dt_vals = filter_ds['time'].values
                dt64_forecast_start = dt_vals[0]
                dt_forecast_start: datetime = pd.to_datetime(dt64_forecast_start)
                dt_forecast_start_str: str = dt_str
                temp_u_column_name: str = f'{dt_forecast_start_str}_u'
                temp_v_column_name: str = f'{dt_forecast_start_str}_v'
                df_dict[temp_u_column_name] = u_vals.reshape(-1)
                df_dict[temp_v_column_name] = v_vals.reshape(-1)
    return df_dict


if __name__ == '__main__':
    read_station_path = r'./../../data_common/stations.csv'
    read_nc_path: str = r'Z:/WIND/GRAPES/2024'
    out_put_path: str = r'E:/01DATA/ML/WIND'
    # step1: 读取站位信息
    stations: List[dict] = read_stations(read_station_path)
    """
        eg:
            [{'code': 'DGG', 'lat': 39.8333, 'lng': 124.1667}, ]
    """
    # step2: 根据站位信息批量读取风场预报时序数据
    files = get_files(read_nc_path)

    # step3: 根据不同站点分别生成该站点对应的预报时序数据
    # step3-1: 获取该年份的 732 组预报时间
    start_arrow: arrow.Arrow = arrow.Arrow(2024, 1, 1, 0, 0, 0, 0, tzinfo=pytz.UTC)
    end_arrow: arrow.Arrow = arrow.Arrow(2024, 1, 1, 0, 0, 0, 0, tzinfo=pytz.UTC)

    time_list = generate_time_range(start_arrow, end_arrow, 12)
    """标准时间集合"""

    # step3-2: 根据标准时间集合 根据指定经纬度获取对应经纬度的指定预报时刻的未来 time_count 预报时刻的预报集合
    for station in stations:
        temp_code: str = station['code']
        temp_lat: float = station['lat']
        temp_lng: float = station['lng']
        # temp_name: str = station['name']
        for temp_time in time_list:
            # 根据 lat 与 lng 读取对应的风场数据
            temp_dict = read_ds_values(read_nc_path, temp_time, temp_code, temp_lat, temp_lng)
            # 根据 temp_time 读取 对应 station 的数据

            temp_df = pd.DataFrame(temp_dict)
            out_put_filename: str = f'out_put_{temp_code}_2024.csv'
            save_path: str = str(pathlib.Path(out_put_path) / out_put_filename)
            temp_df.to_csv(save_path)
            pass

    sorted_files = sorted(files, key=get_wind_sort_key)  # 使用独立函数的清晰写法
    """
        eg:
        [WindowsPath('Z:/WIND/GRAPES/2024/GRAPES_2024010100_240h_UV.nc'), WindowsPath('Z:/WIND/GRAPES/2024/GRAPES_2024010112_240h_UV.nc'), WindowsPath('Z:/WIND/GRAPES/2024/GRAPES_2024010200_240h_UV.nc'),]
    """

    pass
