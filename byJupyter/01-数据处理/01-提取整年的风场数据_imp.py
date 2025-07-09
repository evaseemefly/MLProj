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

# 先从海浪数据中提取出经纬度，时间，风，海浪高度
# 解析单个文件，并存于字典内
from sklearn.preprocessing import StandardScaler

from utils.common import merge_dataframe


def batch_readxmlfiles(read_file: str):
    parser = ET.XMLParser(encoding="iso-8859-5")
    # print(read_file)
    Tree = ET.parse(read_file, parser=parser)
    header = []
    row = []
    root = Tree.getroot()
    # print(root.tag)
    dict_temp = {}
    time_node = root.find('./BuoyageRpt/DateTime')
    # print(time_node.tag)
    dict_temp["time"] = time_node.get("DT")
    location_node = root.find('./BuoyageRpt/BuoyInfo/Location')
    longitude_temp = location_node.get("longitude").replace("Ёф", "").replace("E", "")
    min_temp = longitude_temp.split("Ёу")
    longitude = float(min_temp[0]) + float(min_temp[1]) / 60
    dict_temp["longitude"] = longitude
    latitude_temp = location_node.get("latitude").replace("Ёф", "").replace("N", "")
    min_temp = latitude_temp.split("Ёу")
    latitude = float(min_temp[0]) + float(min_temp[1]) / 60
    dict_temp["latitude"] = latitude
    # print(dict_temp["longitude"])
    # print(dict_temp["latitude"])
    BD_node = root.find('./BuoyageRpt/HugeBuoyData/BuoyData')
    dict_temp["WS"] = BD_node.get("WS")
    dict_temp["YBG"] = BD_node.get("YBG")
    return dict_temp


def batch_readncfiles(read_path: str, lat: float, lng: float, month: int = 1):
    """
        根据指定路径遍历该路径下的所有文件，并读取每个文件中的[0,23]h的时序数据(根据经纬度)
    :param read_path: 读取nc的根目录
    :param lat:
    :param lng:
    :return:
    """

    df_nc: xr.Dataset = None
    nc_path = pathlib.Path(read_path)
    df_dict: dict = {}
    for file in nc_path.iterdir():
        # for file in 'GRAPES_2024010100_240h_UV.nc','GRAPES_2024010112_240h_UV.nc':
        # 修改为按月切分
        if file.is_file():
            # 获取当前文件的 datetime 字符串
            dt_str: str = file.name.split('_')[1]
            dt_arrow = arrow.get(dt_str, 'YYYYMMDDHH')
            temp_month = dt_arrow.month
            if month == temp_month:
                # step1: 拼接成文件全路径
                # file_full_path = nc_path / file
                file_full_path_str: str = str(file)
                # step2: 使用 xarray.open_dataset 打开 netcdf文件
                temp_df: xr.Dataset = xr.open_dataset(file_full_path_str)
                # 注意: 打开的 Dataset 有三个维度,目前只需要按照经纬度提取几个位置的24h内的时序数据
                if temp_df is not None:
                    """
                        Coordinates:
                        * latitude             (latitude) float64 -89.94 -89.81 -89.69 ... 89.81 89.94
                        * longitude            (longitude) float64 0.0 0.125 0.25 ... 359.8 359.9
                        * time                 (time) datetime64[ns] 2024-01-01 ... 2024-01-11
                        Data variables:
                            UGRD_10maboveground  (time, latitude, longitude) float32 ...
                            VGRD_10maboveground  (time, latitude, longitude) float32 ...
                    """
                    # 从该文件中提取指定经纬度的时序数据
                    filter_ds = temp_df.sel(latitude=lat, longitude=lng, method='nearest')
                    # 分别取出 u 与 v 分量
                    u_vals = filter_ds['UGRD_10maboveground'].values[:25]
                    v_vals = filter_ds['VGRD_10maboveground'].values[:25]
                    dt_vals = filter_ds['time'].values
                    dt64_forecast_start = dt_vals[0]
                    dt_forecast_start: datetime = pd.to_datetime(dt64_forecast_start)
                    dt_forecast_start_str: str = dt_forecast_start.strftime('%Y%m%d%H%M%S')
                    temp_u_column_name: str = f'{dt_forecast_start_str}_u'
                    temp_v_column_name: str = f'{dt_forecast_start_str}_v'
                    df_dict[temp_u_column_name] = u_vals
                    df_dict[temp_v_column_name] = v_vals
                    print(f"读取{file_full_path_str}成功")
                else:
                    df_nc = temp_df
    # 将最终的 dict -> pd.DataFrame
    df = pd.DataFrame(df_dict)
    print('生成最终DataFrame ing ')
    return df


def batch_readncfiles_byyears(read_path: str, out_put_path: str, lat: float, lng: float, year: int = 2024,
                              month: Optional[int] = None, count: int = 60):
    """
        根据指定路径遍历该路径下的所有文件，并读取每个文件中的[0,23]h的时序数据(根据经纬度)
    :param read_path: 读取nc的根目录
    :param lat:
    :param lng:
    :return:
    """

    df_nc: xr.Dataset = None
    nc_path = pathlib.Path(read_path)
    df_dict: dict = {}
    # out_put_path: str = r'G:\05DATA\01TRAINING_DATA\WIND'
    for file in nc_path.iterdir():
        # for file in 'GRAPES_2024010100_240h_UV.nc','GRAPES_2024010112_240h_UV.nc':
        # 修改为按月切分
        if file.is_file():
            try:
                # 获取当前文件的 datetime 字符串
                dt_str: str = file.name.split('_')[1]
                dt_arrow = arrow.get(dt_str, 'YYYYMMDDHH')
                temp_year = dt_arrow.year
                temp_month = dt_arrow.month
                if year == temp_year:
                    if month is not None:
                        if temp_month == month:
                            # step1: 拼接成文件全路径
                            # file_full_path = nc_path / file
                            file_full_path_str: str = str(file)
                            # step2: 使用 xarray.open_dataset 打开 netcdf文件
                            temp_df: xr.Dataset = xr.open_dataset(file_full_path_str)
                            # 注意: 打开的 Dataset 有三个维度,目前只需要按照经纬度提取几个位置的24h内的时序数据
                            if temp_df is not None:
                                """
                                    Coordinates:
                                    * latitude             (latitude) float64 -89.94 -89.81 -89.69 ... 89.81 89.94
                                    * longitude            (longitude) float64 0.0 0.125 0.25 ... 359.8 359.9
                                    * time                 (time) datetime64[ns] 2024-01-01 ... 2024-01-11
                                    Data variables:
                                        UGRD_10maboveground  (time, latitude, longitude) float32 ...
                                        VGRD_10maboveground  (time, latitude, longitude) float32 ...
                                """
                                # 从该文件中提取指定经纬度的时序数据
                                filter_ds = temp_df.sel(latitude=lat, longitude=lng, method='nearest')
                                # 分别取出 u 与 v 分量
                                u_vals = filter_ds['UGRD_10maboveground'].values[:count]
                                v_vals = filter_ds['VGRD_10maboveground'].values[:count]
                                # TODO:[-] 25-05-21 若数据有空缺，对数据进行填充
                                u_vals_padded = np.pad(u_vals, (0, count - len(u_vals)), constant_values=np.nan)
                                v_vals_padded = np.pad(v_vals, (0, count - len(v_vals)), constant_values=np.nan)

                                dt_vals = filter_ds['time'].values
                                dt64_forecast_start = dt_vals[0]
                                dt_forecast_start: datetime = pd.to_datetime(dt64_forecast_start)
                                dt_forecast_start_str: str = dt_forecast_start.strftime('%Y%m%d%H%M%S')
                                temp_u_column_name: str = f'{dt_forecast_start_str}_u'
                                temp_v_column_name: str = f'{dt_forecast_start_str}_v'
                                df_dict[temp_u_column_name] = u_vals_padded
                                df_dict[temp_v_column_name] = v_vals_padded
                                print(f"读取{file_full_path_str}成功")
                            else:
                                df_nc = temp_df
            except Exception as e:
                print(e.args)
    # 将最终的 dict -> pd.DataFrame
    df = pd.DataFrame(df_dict)
    print('生成最终DataFrame ing ')
    out_put_filename: str = f'out_put_MF01001_{str(year)}_{str(month)}.csv'
    save_path: str = str(pathlib.Path(out_put_path) / out_put_filename)
    df.to_csv(save_path)
    print(f'存储路径:{save_path}')
    # return df


def merge_warmup_dataset(read_path: str, out_path: str, lat: float, lng: float, year: int = 2024,
                         month: Optional[int] = None, count: int = 60, warmup_count=4) -> pd.DataFrame:
    """
        加入对前置数据的预热，并合并
    :param read_path:
    :param out_path:
    :param lat:
    :param lng:
    :param year:
    :param month:
    :param count:
    :param warmup_count:
    :return:
    """
    files = get_files(read_path)

    sorted_files = sorted(files, key=get_sort_key)  # 使用独立函数的清晰写法
    # 遍历文件集合，根据当前文件获取上一个预报时次的文件，并按时间提取并校正后进行拼接，输出

    df: pd.DataFrame = None
    list_df: List[pd.DataFrame] = []

    start_index = 1
    """起始下标——从index=1开始"""

    # TODO:[*] 25-07-03 只取 10 个文件
    # sorted_files = sorted_files[:10]

    if len(sorted_files) < 2:
        print('文件数不满足条件，跳出')
    else:
        for previous_file, next_file in zip(sorted_files, sorted_files[start_index:]):
            # 在这里执行您的处理逻辑
            print(f"正在处理文件对:")
            print(f"  前一个文件: {previous_file.name}")
            print(f"  后一个文件: {next_file.name}")
            previous_file_path: pathlib.Path = pathlib.Path(read_path) / previous_file
            next_file_path: pathlib.Path = pathlib.Path(read_path) / next_file
            col_name, current_abs_df = batch_readncfiles2warmup_byears(previous_file_path, next_file_path, out_path,
                                                                       lat,
                                                                       lng, 2024)
            temp_df = current_abs_df[col_name]
            list_df.append(temp_df)
            print("-" * 20)
            pass

    if len(list_df) > 0:
        df = pd.concat(list_df, axis=1)
    return df


def batch_readncfiles2warmup_byears(previous_file_path: pathlib.Path, current_file_path: pathlib.Path,
                                    out_put_path: str,
                                    lat: float, lng: float,
                                    year: int = 2024,
                                    month: Optional[int] = None, count: int = 60, warmup_len: int = 4) -> tuple[
    str, pd.DataFrame]:
    """
        + 25-07-01 加入数据预热的流程
        根据指定路径遍历该路径下的所有文件，并读取每个文件中的[0,23]h的时序数据(根据经纬度)
        获取包含上一个预报发布时刻的 warmup_len 长度的预热数据的 u 与 v 分量的 Dataframe
    :param previous_file_path:
    :param current_file_path:
    :param out_put_path:
    :param lat:
    :param lng:
    :param year:
    :param month:
    :param count:
    :param warmup_len:
    :return:
    """

    df_nc: xr.Dataset = None
    previous_file_path_str: str = str(previous_file_path)
    current_file_path_str: str = str(current_file_path)

    u_data_dict = {}
    v_data_dict = {}
    abs_data_dict = {}
    data_dict = {}
    # 获取当前文件的 datetime 字符串
    dt_str: str = current_file_path.name.split('_')[1]
    dt_arrow = arrow.get(dt_str, 'YYYYMMDDHH')
    # 提取前一个预报时刻的头 num 个时次的值

    # step2: 使用 xarray.open_dataset 打开 netcdf文件
    previous_df: xr.Dataset = xr.open_dataset(previous_file_path_str)
    """前一个预报发布时刻的预报结果"""
    current_df: xr.Dataset = xr.open_dataset(current_file_path_str)
    """当前预报发布时刻的预报结果"""
    # 注意: 打开的 Dataset 有三个维度,目前只需要按照经纬度提取几个位置的24h内的时序数据
    if previous_df is not None and current_df is not None:
        """
            Coordinates:
            * latitude             (latitude) float64 -89.94 -89.81 -89.69 ... 89.81 89.94
            * longitude            (longitude) float64 0.0 0.125 0.25 ... 359.8 359.9
            * time                 (time) datetime64[ns] 2024-01-01 ... 2024-01-11
            Data variables:
                UGRD_10maboveground  (time, latitude, longitude) float32 ...
                VGRD_10maboveground  (time, latitude, longitude) float32 ...
        """

        # step2-1 前置预报数据，取出预热数据长度
        # 从该文件中提取指定经纬度的时序数据
        previous_filter_ds = previous_df.sel(latitude=lat, longitude=lng, method='nearest')
        # 分别取出 u 与 v 分量
        # TODO:[-] 25-07-09 注意此处 value[start:end] => [start,end)
        previous_u_vals: np.ndarray = previous_filter_ds['UGRD_10maboveground'].values[:warmup_len + 1]
        previous_v_vals: np.ndarray = previous_filter_ds['VGRD_10maboveground'].values[:warmup_len + 1]

        dt_vals = previous_df['time'].values

        # step2-2 当前预报时段数据，取出 count 长度
        # 从该文件中提取指定经纬度的时序数据
        current_filter_ds = current_df.sel(latitude=lat, longitude=lng, method='nearest')
        # 分别取出 u 与 v 分量
        current_u_vals = current_filter_ds['UGRD_10maboveground'].values[:count]
        current_v_vals = current_filter_ds['VGRD_10maboveground'].values[:count]

        # step2: 计算偏差
        #
        bias_u = current_u_vals[0] - previous_u_vals[warmup_len]
        bias_v = current_v_vals[0] - previous_v_vals[warmup_len]

        # step3: 计算平滑后的 u 与 v
        previous_u_smoothing_vals = previous_u_vals + bias_u
        previous_v_smoothing_vals = previous_v_vals + bias_v
        # TODO:[-] 25-07-09 previous_u_smoothing_vals[最后一个值] 应等于 current_u_vals[0]
        current_u_vals = current_u_vals[1:count]
        current_v_vals = current_v_vals[1:count]

        # step4: 将前值置于 current 之前
        # all_u_smoothing_vals = pd.concat([previous_u_smoothing_vals, current_u_vals])
        # all_v_smoothing_vals = pd.concat([previous_v_smoothing_vals, current_v_vals])
        all_u_smoothing_vals = np.concatenate((previous_u_smoothing_vals, current_u_vals), axis=0)
        all_v_smoothing_vals = np.concatenate((previous_v_smoothing_vals, current_v_vals), axis=0)
        all_abs_smoothing_vals = np.sqrt(all_u_smoothing_vals ** 2 + all_v_smoothing_vals ** 2)

        # step3:
        u_data_dict[dt_str] = all_u_smoothing_vals
        v_data_dict[dt_str] = all_v_smoothing_vals
        abs_data_dict[dt_str] = all_abs_smoothing_vals

        data_dict[f'{dt_str}_u'] = all_u_smoothing_vals
        data_dict[f'{dt_str}_v'] = all_v_smoothing_vals
        data_dict[f'{dt_str}_abs'] = all_abs_smoothing_vals
    # return pd.DataFrame.from_dict(u_data_dict), pd.DataFrame.from_dict(v_data_dict)
    # return df
    return f'{dt_str}_abs', pd.DataFrame.from_dict(data_dict)


def batch_get_realdata(file_full_path: str, split_hours=72, issue_hours_steps: int = 12) -> pd.DataFrame:
    """
        TODO:[-] 25-04-23 生成实况训练数据集
        从指定文件批量获取时间数据并以dataframe的形式返回
    :param file_full_path:
    :return:
    """

    """
        eg: csv文件样例:
                        time	longitude	latitude	WS	YBG
                        202401010000
                        YYYYMMDDHHmm
    """
    list_series = []
    merge_dict = {}
    if pathlib.Path(file_full_path).exists():
        # ds: xr.Dataset = xr.open_dataset(file_full_path)
        df: pd.DataFrame = pd.read_csv(file_full_path)
        """读取指定路径的浮标处理后的一年的数据"""
        # 通过起止时间找到对应的index，然后每次的发布时间间隔步长为12h

        # step1: 生成2024年一年的时间步长为1hour的时间索引集合
        start_time = '2024-01-01 00:00:00'
        end_time = '2024-12-31 23:00:00'
        time_series = pd.date_range(start=start_time, end=end_time, freq='H')

        # 将time列的内容从int64 => str
        df['time'] = df['time'].astype(str)
        df['time'] = pd.to_datetime(df['time'], format='%Y%m%d%H%M')
        # step2: 将 time列设置为index，并将index替换为标准时间集合
        df.set_index('time', inplace=True)
        df_reindexed = df.reindex(time_series)
        df_reindexed.index.name = 'time'

        # step3: 生成12小时为间隔的时间数组
        freq_str: str = f'{issue_hours_steps}H'
        issue_dt_series = pd.date_range(start=start_time, end=end_time, freq=freq_str)

        for temp_time in issue_dt_series:
            temp_index: int = df_reindexed.index.get_loc(temp_time)
            val_series = df_reindexed[temp_index:temp_index + split_hours]
            list_series.append(val_series)
        # TODO:[-] 25-04-24 此处做重新修改，拼接成一个dataframe

        for temp_time in issue_dt_series:
            dt_str: str = temp_time.strftime('%Y%m%d%H%M%S')
            temp_index: int = df_reindexed.index.get_loc(temp_time)
            val_series = df_reindexed[temp_index:temp_index + split_hours]
            # 此处改为只取 'WS' 列
            # TODO:[-] 25-04-24 住一次此处需要将每一个 series的index索引重置为 [0,71]
            merge_dict[dt_str] = val_series['WS'].reset_index(drop=True)
            # list_series.append(val_series)
    df = pd.DataFrame.from_dict(merge_dict)
    return df


def get_test_array(test_read_path: str, training_read_path: str, issue_times_index: DatetimeIndex):
    """
        分别读取测试数据集以及实况数据集并进行训练
    :param test_read_path:
    :param training_read_path:
    :return:
    """
    if pathlib.Path(test_read_path).exists() and pathlib.Path(training_read_path).exists():
        df_test: pd.DataFrame = pd.read_csv(test_read_path)
        u_data_dict = {}
        v_data_dict = {}
        # 读取的预报风场——测试训练集 在 df 中是通过 xxx_u与 xxx_v 的形式进行存储
        # TODO:[-] 25-04-28 u 与 v 每个共613组预报数据
        for col_name in df_test.columns:
            try:
                col_vector = df_test[col_name]
                # yyyymmddhhss
                dt_temp_str: str = col_name.split('_')[0]
                # u or v
                var_temp_str: str = col_name.split('_')[1]
                if var_temp_str == 'u':
                    # u_data_dict[dt_temp_str] = col_vector.tolist()
                    u_data_dict[dt_temp_str] = col_vector
                elif var_temp_str == 'v':
                    # v_data_dict[dt_temp_str] = col_vector.tolist()
                    v_data_dict[dt_temp_str] = col_vector
                print(f'当前列:{col_name}处理成功~')
            except Exception as e:
                print(f'当前列:{col_name}处理错误!')
        # # step2: 将字典统一转换为二维数组
        # result_u_array = [val for key, val in u_data_dict.items()]
        # result_v_array = [val for key, val in v_data_dict.items()]
        # return [result_u_array, result_v_array]
        df_u = pd.DataFrame.from_dict(u_data_dict)
        df_v = pd.DataFrame.from_dict(v_data_dict)
        # 将时间字符串=>datetime
        df_u.columns = pd.to_datetime(df_u.columns)
        df_v.columns = pd.to_datetime(df_v.columns)
        # TODO:[*] 25-04-29
        # 需要根据起止时间及时间步长，生成对应的时间索引，并将该时间索引作为标准索引
        # 注意： reindex 后会返回一个新的 DataFrame，并不会修改原始df
        df_u = df_u.reindex(columns=issue_times_index)
        df_v = df_v.reindex(columns=issue_times_index)
        return df_u, df_v
        # pass
    return None


def get_files(read_path: str):
    """
        获取指定路径下的所有文件
    :param read_path:
    :return:
    """
    files_name = [temp for temp in pathlib.Path(read_path).iterdir()]
    files = []
    for file_temp in files_name:
        if file_temp.is_file():
            files.append(file_temp)
    return files


def get_sort_key(file_path):
    """从文件名中提取 YYYYMMDDHH 部分作为排序键"""
    # 这是一个更健壮的写法，以防文件名结构略有不同
    try:
        # 文件名示例: GRAPES_2024010100_240h_UV.nc
        # 分割后: ['GRAPES', '2024010100', '240h', 'UV.nc']
        date_time_str = file_path.name.split('_')[1]
        return date_time_str
    except IndexError:
        # 如果文件名不符合预期格式，返回一个空字符串，使其排在最前或最后
        return ""


def main():
    """
        TODO:[-] 25-06-08 注意由于风场数据提起手动生成的时间轴索引原始为:1hour => 3hour
    :return:
    """

    # read_path: str = r'/Volumes/DATA/WIND/GRAPES/2024'
    # # read_path: str = r'/Volumes/DATA/WIND/TEST/2024'
    # out_put_path: str = r'/Volumes/DATA/01TRAINNING_DATA/WIND/01'
    # out_put_file_path: str = str(pathlib.Path(out_put_path) / 'GRAPES_2024_24')
    # TODO:[-] 25-06-09 razer 配置
    read_path: str = r'Z:/WIND/GRAPES/2024'
    out_put_path: str = r'Z:/SOURCE_DATA/2024_warmup_dataset_250709.csv'
    lat: float = 39.5003
    lng: float = 120.59533
    # TODO:[-] 25-06-30 加入数据预热
    files = get_files(read_path)

    sorted_files = sorted(files, key=get_sort_key)  # 使用独立函数的清晰写法

    # step1: 提取2024年的风场数据，并加入warm_up——预热数据
    # 将文件降序排列
    df_merge = merge_warmup_dataset(read_path, out_put_path, lat, lng, 2024)
    df_merge.to_csv(out_put_path)

    # @expired—— 250707
    # step1:批量读取nc文件并提取指定经纬度的72小时数据并拼接成 dataframe
    # for index in np.arange(0, 12):
    #     temp_month = index + 1
    #     batch_readncfiles_byyears(read_path, out_put_path, lat, lng, 2024, temp_month, 72)

    # step2:
    # step2:将上面step1处理的按月保存的72小时浮标站位的时序数据合并为一整年的数据
    merge_df = merge_dataframe(out_put_path)
    # merge_full_path: str = str(pathlib.Path(out_put_path) / 'merge.csv')
    # merge_df.to_csv(merge_full_path)

    # -------------

    pass


if __name__ == '__main__':
    main()
