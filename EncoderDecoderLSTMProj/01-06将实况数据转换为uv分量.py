import csv
import pathlib
import shutil
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import arrow

READ_DIR_PATH: pathlib.Path = pathlib.Path(r'E:\01DATA\ML\STATION_CONVERT_WS')
READ_PATH: pathlib.Path = pathlib.Path(r'E:\01DATA\ML\STATION_CONVERT_WS\鲅鱼圈_standard.csv')
SAVE_PATH: pathlib.Path = pathlib.Path(r'E:\01DATA\ML\STATION_CONVERT_SORTED_WS')

if __name__ == '__main__':
    for source_file in READ_DIR_PATH.glob('*.csv'):
        source_file_name: str = source_file.name
        df = pd.read_csv(source_file)
        # 获取所有columns
        times_list: List[str] = [col for col in df.columns if '_' in col]
        uv_data: Dict[str, pd.Series] = {}
        # 排序
        sorted_times_list = sorted(times_list)
        # 对time str 去重
        time_sets = set([col.split('_')[0] for col in sorted_times_list])

        # 遍历获取 对应时间的 ws 与 wd，并将(ws,wd)=>(u,v)
        # TODO:[*] 25-08-26 此处处理后 columns 排序有误
        for time_set in time_sets:
            source_wd_stamp: str = f'{time_set}_wd'
            source_ws_stamp: str = f'{time_set}_ws'
            wd_list = df[source_wd_stamp]
            ws_list = df[source_ws_stamp]
            # 1. 将风向（角度制）转换为弧度制
            # np.radians() 可以处理整个 Pandas Series
            direction_rad = np.radians(wd_list)

            # 2. 应用公式计算 u 和 v 分量
            # NumPy的sin和cos函数同样支持向量化操作
            u = -ws_list * np.sin(direction_rad)
            v = -ws_list * np.cos(direction_rad)

            # 3. 将计算出的 u, v 分量作为新列添加回 DataFrame
            # 这样可以保持数据结构的一致性
            target_u_stamp: str = f'{time_set}_u'
            target_v_stamp: str = f'{time_set}_v'
            uv_data[target_u_stamp] = u
            uv_data[target_v_stamp] = v
        uv_df = pd.DataFrame(uv_data)
        # 需要对 columns 进行排序
        time_cols = [col for col in uv_df.columns if '_' in col]
        sorted_time_cols = sorted(time_cols)
        df_uv_sorted = uv_df[sorted_time_cols]
        save_full_path: str = str(SAVE_PATH / f'{source_file_name}_uv_sorted.csv')
        df_uv_sorted.to_csv(save_full_path, index=True)
    pass
