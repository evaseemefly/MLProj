import csv
import pathlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import xarray as xr
import arrow
from exceptiongroup import catch

from private.dicts import dicts_station


def batch_extract_wind(read_path: pathlib.Path, lat: float, lng: float):
    """
        基于 lat与lng，从 read_path 文件读取 对应经纬度的时序数据集
        生成的数据集为
        index           U   |  V
        2025010100   |      |
        2025010103   |      |
        2025010106   |      |
        .........
    :param read_path:
    :return:
    """
    df_dict: dict = {}
    source_ds: xr.Dataset = xr.open_dataset(read_path)
    filter_ds = source_ds.sel(latitude=lat, longitude=lng, method='nearest')
    time_count: int = 25
    # 分别取出 u 与 v 分量
    u_vals = filter_ds['UGRD_10maboveground'].values[:time_count]
    v_vals = filter_ds['VGRD_10maboveground'].values[:time_count]
    dt_vals = filter_ds['time'].values[:time_count]
    dt64_forecast_start = dt_vals[0]
    # 根据起始时间获取对应的 time 索引
    # arrow.get(dt64_forecast_start)
    dt_forecast_start: datetime = pd.to_datetime(dt64_forecast_start)

    dt_forecast_start_str: str = dt_forecast_start.strftime('%Y%m%d%H%M%S')
    temp_u_column_name: str = f'{dt_forecast_start_str}_u'
    temp_v_column_name: str = f'{dt_forecast_start_str}_v'
    # df_dict[temp_u_column_name] = u_vals
    # df_dict[temp_v_column_name] = v_vals
    # 行向量 => 列向量
    df_dict[temp_u_column_name] = u_vals.reshape(-1)
    df_dict[temp_v_column_name] = v_vals.reshape(-1)

    df = pd.DataFrame(df_dict)
    # [0,25)
    index_list = pd.RangeIndex(start=0, stop=time_count, step=1)
    df.set_index(index_list, inplace=True)
    return df


def get_wind_files(read_path: pathlib.Path) -> List[pathlib.Path]:
    """
        获取指定目录下的所有风场文件
    :param read_path:
    :return:
    """
    files: List[pathlib.Path] = []
    if not read_path.exists():
        raise FileNotFoundError(read_path)
    for source_file in read_path.rglob('*.nc'):
        files.append(source_file)
    return files


def merge_station_dataframe(source_path: pathlib.Path, target_path: pathlib.Path):
    wind_files = get_wind_files(source_path)
    # wind_files = wind_files[:5]
    station: List[dict] = [
        # {'code': 'DGG', 'lat': 39.8333, 'lng': 124.1667},
        # {'code': 'LHT', 'lat': 38.8667, 'lng': 121.6833},
        # {'code': 'BYQ', 'lat': 40.3, 'lng': 122.0833},
        # {'code': 'HLD', 'lat': 40.7167, 'lng': 121},
        {'code': 'QHD', 'lat': 39.9167, 'lng': 119.6167},
        {'code': 'TGU', 'lat': 38.9378, 'lng': 117.8261},
        {'code': 'CFD', 'lat': 38.9333, 'lng': 118.5},
        {'code': 'HHA', 'lat': 38.3167, 'lng': 117.8667},
        {'code': 'WFG', 'lat': 37.2333, 'lng': 119.1833},
        {'code': 'PLI', 'lat': 37.8333, 'lng': 120.6167},
        {'code': 'CST', 'lat': 37.3833, 'lng': 122.7}, ]

    for station in station:
        temp_code: str = station['code']
        df_list = []
        print('------------------------------------')
        for wind_file in wind_files:
            try:

                temp_df = batch_extract_wind(wind_file, station.get('lat'), station.get('lng'))
                df_list.append(temp_df)
                print(f'[-]提取:f{str(wind_file)}文件成功~')
                pass
            except Exception as e:
                print(f'[*]提取:f{str(wind_file)}文件失败!')
        #  axis=1 按列拼接; 默认是 axis=0，按行拼接
        df_merged = pd.concat(df_list, axis=1)
        temp_save_name: str = f'{station.get("code")}_2024_uv.csv'
        temp_save_path: pathlib.Path = pathlib.Path(target_path) / temp_save_name
        df_merged.to_csv(str(temp_save_path), index=True)
        print(f'[!]提取{temp_code}站点文件成功!')
        print('------------------------------------')
    pass


def sorted_station_merge_dataset(full_path: pathlib.Path) -> xr.Dataset:
    """
        按照时间升序排序
    :param full_path:
    :return:
    """
    df = pd.read_csv(full_path)
    time_cols = [col for col in df.columns if '_' in col]
    sorted_time_columns = sorted(time_cols)
    df_sorted = df[sorted_time_columns]
    source_name: str = full_path.name.split('.')[0]
    saved_path: pathlib.Path = full_path.parent / f'{source_name}_sorted.csv'
    df_sorted = df_sorted.to_csv(str(saved_path))

    pass


if __name__ == '__main__':
    WIND_SOURCE_ROOT: str = r'Z:\WIND\GRAPES\2024'
    WIND_SAVE_ROOT: str = r'E:\01DATA\ML\WIND_STATIONS'
    WIND_SORTED_ROOT: str = r'E:\01DATA\ML\WIND_STATIONS_SORTED'
    WIND_SOURCE_PATH: pathlib.Path = pathlib.Path(WIND_SOURCE_ROOT)
    WIND_SAVE_PATH: pathlib.Path = pathlib.Path(WIND_SAVE_ROOT)
    WIND_SORTED_PATH: pathlib.Path = pathlib.Path(WIND_SAVE_ROOT)
    # step1:
    # 批量根据 code 拼接 2024年全年的海洋站数据(按照站点code进行拼接)
    # merge_station_dataframe(WIND_SOURCE_PATH, WIND_SAVE_PATH)

    # step2:
    # 生成排序后的 dataframe
    saved_files = [source_file for source_file in WIND_SORTED_PATH.rglob('*.csv')]
    for file_temp in saved_files:
        sorted_station_merge_dataset(file_temp)
    pass
