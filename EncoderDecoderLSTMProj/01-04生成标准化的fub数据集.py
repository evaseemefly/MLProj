import pathlib
from typing import List

from arrow import arrow
# TODO:[-] 25-08-05 注意使用lxml库，不使用xml库，lxml可以更好的处理encoder的问题
# import xml.etree.ElementTree as ET
from lxml import etree as ET
import pandas as pd
import numpy as np
from tqdm import tqdm  # 引入tqdm库，用于显示进度条，让等待过程更直观

from utils.common import dms2decimal

fub_codes = ['MF01001', 'MF01002', 'MF01004', 'MF02004', 'MF02001']
"""浮标code集合"""


def main():
    code: str = 'MF02004'
    dir_path: str = r'E:\01DATA\ML\FUB'
    out_put_path: str = r'E:\01DATA\ML\FUB_RESAMPLED'
    year = 2024
    data_dir_path = pathlib.Path(dir_path)

    fub_source_list = list(pathlib.Path(dir_path).rglob('*.csv'))
    for fub_file in fub_source_list:
        df_temp: pd.DataFrame = pd.read_csv(str(fub_file))
        # 步骤 1: 将 'DT' 列转换为标准日期时间格式
        # '%Y%m%d%H' 是解析 'yyyymmddHH' 格式的关键
        df_temp['DT'] = pd.to_datetime(df_temp['DT'], format='%Y%m%d%H%M')
        # 步骤 2: 将转换后的 'DT' 列设置为 DataFrame 的索引
        df_temp = df_temp.set_index('DT')

        if df_temp.index.has_duplicates:
            print(f'!!{fub_file.name}存在重复的时间索引项!!')
            # TODO:[-] 25-08-06 解决可能存在重复row的错误
            # 通过判断索引是否重复来过滤数据。
            # ~df_temp.index.duplicated() 会保留每个重复项中的第一个。
            # keep='first' 是默认行为，也可以明确指定。
            df_temp = df_temp[~df_temp.index.duplicated(keep='first')]

        # 创建时间索引列
        # 步骤 3: 创建一个从 2024 年初到年末的完整小时级索引
        # 注意：2024年是闰年，有366天
        full_hourly_index = pd.date_range(
            start='2024-01-01 00:00:00',
            end='2024-12-31 23:00:00',
            freq='H',  # 'H' 表示每小时 (Hourly)
            name='DT'  # 为索引命名，保持列名一致
        )

        # 步骤 4: 使用新索引对 DataFrame 进行 reindex，缺失值将自动填充为 NaN
        df_resampled = df_temp.reindex(full_hourly_index)

        source_file_name: str = fub_file.name.split('.')[0]
        #
        file_resampled_name: str = f'{source_file_name}_resampled.csv'
        full_convert_file_path: str = str(pathlib.Path(out_put_path) / file_resampled_name)
        df_resampled.to_csv(full_convert_file_path, index=True, encoding='utf-8-sig')
        print(f'输出:{full_convert_file_path}成功!')
        print("-" * 30)
    pass


if __name__ == '__main__':
    main()
    pass
