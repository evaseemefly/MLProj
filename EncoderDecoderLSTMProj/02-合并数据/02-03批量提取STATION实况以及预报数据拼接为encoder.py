from typing import List, Optional

import arrow
from pathlib import Path
import pytz
import numpy as np
import pandas as pd


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


def generate_time_range_bystep(start: arrow.Arrow, count: int, step_hour: int = 3) -> List[arrow.Arrow]:
    """
        根据起始时间，以及长度，生成等差时间数列(hour)
    :param start:
    :param count:
    :param step_hour:
    :return:
    """
    generate_arrows: List[arrow.Arrow] = []
    while len(generate_arrows) < count:
        generate_arrows.append(start)
        start = start.shift(hours=step_hour)
    return generate_arrows


def main():
    read_path: Path = Path(r'E:\01DATA\ML\WIND_STATIONS_SORTED')
    realdata_allyear_path: Path = Path(r'E:\01DATA\ML\海洋站数据处理\station_allyear_utc_uv')
    mergedata_h5_path: Path = Path(r'E:\01DATA\ML\MERGEDATA_H5')
    start_arrow: arrow.Arrow = arrow.Arrow(2024, 1, 1, 0, 0, 0, 0, tzinfo=pytz.UTC)
    end_arrow: arrow.Arrow = arrow.Arrow(2025, 1, 1, 0, 0, 0, 0, tzinfo=pytz.UTC)

    times = generate_time_range(start_arrow, end_arrow, step_hour=12)

    for temp_file in read_path.rglob('*.csv'):
        # 取 0 -23 行数据
        temp_df: pd.DataFrame = pd.read_csv(temp_file, nrows=24)
        """风场的 24 个预报时间数据(3hour)"""
        temp_code: str = temp_file.name.split('.')[0].split('_')[0]
        """当前遍历的站点code"""

        temp_code_h5_filename: str = f'2024_{temp_code}_mergedata.h5'

        temp_code_h5_fullpath: Path = Path(mergedata_h5_path) / temp_code_h5_filename

        # 批量根据经纬度从风场中提取预报数据


        # 获取 dataframe 中的所有时间戳
        temp_forecast_timestamps = list(set([temp_col.split('_')[0] for temp_col in temp_df.columns[1:]]))
        # 排序
        temp_forecast_timestamps: List[str] = sorted(temp_forecast_timestamps)
        # 按照12小时时间间隔获取对应的预报集合，并根据预报时间生成提取实况的时间索引表

        temp_station_mergedata_dict: dict = {}

        # 改为使用 dataframe 中 columns 对应的时间str拼接实况数据
        for temp_time in temp_forecast_timestamps:
            # for temp_time in times:
            # 20240101000000_u
            # 202401000000_u
            temp_time: arrow.Arrow = arrow.get(temp_time, 'YYYYMMDDHHmmss')
            temp_time_str: str = temp_time.format('YYYYMMDDHHmmSS')
            temp_col_u_name: str = f'{temp_time_str}_u'
            temp_col_v_name: str = f'{temp_time_str}_v'
            temp_forecast_v_series = temp_df[temp_col_v_name]
            temp_forecast_v_series.name = 'forecast_v'
            temp_forecast_u_series = temp_df[temp_col_u_name]
            temp_forecast_u_series.name = 'forecast_u'
            temp_forecast_df: pd.DataFrame = pd.concat([temp_forecast_v_series, temp_forecast_u_series], axis=1)
            realdata_times_index_arrow = generate_time_range_bystep(temp_time, 24, 3)
            realdata_dt_index = [t.datetime for t in realdata_times_index_arrow]
            realdata_time_index = pd.to_datetime(realdata_dt_index)
            temp_forecast_df.set_index(realdata_time_index, inplace=True)

            merge_df: Optional[pd.DataFrame] = None

            # eg: WFG_uv.csv
            temp_station_realdata_name: str = f'{temp_code}_uv.csv'
            temp_station_realdata_full_path: Path = realdata_allyear_path / temp_station_realdata_name
            if temp_station_realdata_full_path.exists():
                try:
                    # print(f'reading {str(temp_station_realdata_full_path)} file ing')
                    temp_df_realdata = pd.read_csv(str(temp_station_realdata_full_path), index_col=0)
                    temp_df_realdata = temp_df_realdata[['u', 'v']]
                    # 对 U 与 v 列重命名
                    temp_df_realdata.rename(columns={'u': 'realdata_u', 'v': 'realdata_v'},
                                            inplace=True)
                    # 转换 dataframe 的 index
                    temp_df_realdata.index = pd.to_datetime(temp_df_realdata.index)
                    # 将 df 与 time_index 索引求交际
                    temp_selected_time_index = temp_df_realdata.index.intersection(realdata_time_index)
                    temp_df_realdata = temp_df_realdata.loc[temp_selected_time_index]
                    """根据指定的时间索引——start起止时间，step 为 3hour——按行筛选"""

                    merge_df = pd.concat([temp_forecast_df, temp_df_realdata], axis=1)
                    pass
                except FileNotFoundError as exception:
                    print(f'{str(temp_station_realdata_full_path)} not exist')
                    print(exception.args)
                pass
            temp_station_mergedata_dict[temp_time_str] = merge_df
            pass
        try:
            # TODO:[*] 25-09-11 ERROR: ImportError("Missing optional dependency 'pytables'.  Use pip or conda to install pytables.")
            with pd.HDFStore(str(temp_code_h5_fullpath), mode='w') as store:
                for key, df in temp_station_mergedata_dict.items():
                    store.put(key, df, format='table', data_columns=True)

            print(f"数据已保存到 {str(temp_code_h5_fullpath)}")
        except Exception as exception:
            print(f'{str(temp_code_h5_fullpath)} error: {str(exception)}')
    pass


if __name__ == '__main__':
    main()
    pass
