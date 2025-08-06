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
    dir_path: str = r'Z:\FUB\处理后'
    out_put_path: str = r'E:\01DATA\ML\FUB'
    year = 2024
    data_dir_path = pathlib.Path(dir_path)

    fub_source_list = list(pathlib.Path(dir_path).rglob('*.csv'))
    for fub_file in fub_source_list:
        df = pd.read_csv(fub_file)
        df['lon'] = df['longitude'].apply(dms2decimal)
        df['lat'] = df['latitude'].apply(dms2decimal)
        print("\n转换后的 DataFrame:")
        print(df)
        print("-" * 30)
        # 4. 保留小数后6位
        df['lon'] = df['lon'].round(6)
        df['lat'] = df['lat'].round(6)
        source_file_name: str = fub_file.name.split('.')[0]
        convert_file_name: str = f'{source_file_name}_convert.csv'
        full_convert_file_path: str = str(pathlib.Path(out_put_path) / convert_file_name)
        df.to_csv(full_convert_file_path, index=False, encoding='utf-8-sig')
        print(f'输出:{full_convert_file_path}成功!')
        print("-" * 30)
    pass


if __name__ == '__main__':
    main()
    pass
