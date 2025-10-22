from typing import List

import pandas as pd
from pathlib import Path

import pytz
from arrow import Arrow


def generate_time_range(start_arrow, end_arrow, step_hour) -> List[Arrow]:
    # 2. 初始化一个空列表用于存放结果
    result_arrows = []

    # 3. 初始化当前时间为起始时间
    current_arrow = start_arrow

    # 4. 循环直到当前时间超过结束时间
    while current_arrow <= end_arrow:
        # 将当前时间添加到结果列表中
        result_arrows.append(current_arrow)
        # 将当前时间增加12小时，作为下一次循环的时间点
        current_arrow = current_arrow.shift(hours=step_hour)

    return result_arrows


def main(file_path: Path):
    start_arrow, end_arrow = Arrow(2024, 1, 1, 0, 0, 0, tzinfo=pytz.UTC), Arrow(2025, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)
    standard_arrows = generate_time_range(start_arrow, end_arrow, step_hour=12)
    # '/2024 01 01 00 00 00'
    standard_dt_str = [temp.format('YYYYMMDDHHmmss') for temp in standard_arrows]
    set_standard_dt_str = set(standard_dt_str)
    try:
        # 使用 HDFStore 以只读模式('r')打开文件
        with pd.HDFStore(str(file_path), mode='r') as store:
            print("文件中的所有 keys 如下:")
            time_keys = [temp[1:] for temp in store.keys()]
            set_time_keys = set(time_keys)
            diff_time_str = list(set_standard_dt_str - set_time_keys)
            diff_time_str = sorted(diff_time_str)
            print(diff_time_str)
            print(f'读取的store中的keys的长度为:{len(diff_time_str)}')
            pass



    except Exception as e:
        print(f"错误：文件 '{str(file_path)}' 未找到。")
        print(e)
        pass


if __name__ == '__main__':
    file_path: Path = Path(r'E:\01DATA\ML\MERGEDATA_H5\STATIONS\2024_BYQ_mergedata.h5')
    file_fub_merge_path: Path = Path(r'E:\01DATA\ML\MERGEDATA_H5\FUB\2024_MF02001_mergedata.h5')
    main(file_path)
    pass
