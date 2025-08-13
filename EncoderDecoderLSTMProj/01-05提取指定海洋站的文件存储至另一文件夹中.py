import csv
import pathlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def copy_files_with_pathlib(source_dir, target_dir):
    """
    使用 pathlib 将源目录的所有子目录中的文件，全部拷贝到目标目录。
    如果遇到同名文件，会自动重命名以避免覆盖。

    :param source_dir: 源目录路径 (字符串或 Path 对象)
    :param target_dir: 目标目录路径 (字符串或 Path 对象)
    """
    # 1. 将输入路径转换为 Path 对象，这是 pathlib 的核心
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # 2. 检查源目录是否存在
    if not source_path.is_dir():
        print(f"错误：源目录 '{source_path}' 不存在或不是一个目录。")
        return

    if not target_path.exists():
        target_path.mkdir(parents=True, exist_ok=True)

    # 3. 检查并创建目标目录
    # parents=True 相当于 mkdir -p，会创建所有必需的父目录
    # exist_ok=True 表示如果目录已存在，则不抛出错误
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"目标目录 '{target_path}' 已准备就绪。")
    print("-" * 30)

    copied_count = 0
    skipped_count = 0
    renamed_count = 0

    # 4. 递归遍历源目录中的所有文件
    # source_path.rglob('*') 会递归地查找所有匹配项（文件和目录）
    for source_file in source_path.rglob('*'):
        # 确保我们只处理文件，跳过目录
        if source_file.is_file():
            # 5. 构建目标文件路径，使用 '/' 操作符，非常直观
            target_file_path = target_path / source_file.name

            # 6. 【核心】处理同名文件
            if target_file_path.exists():
                # 使用 .stem (文件名) 和 .suffix (扩展名) 属性来构建新名字
                base = target_file_path.stem
                suffix = target_file_path.suffix
                i = 1
                # 循环直到找到一个不存在的新文件名
                while target_file_path.exists():
                    new_name = f"{base}_{i}{suffix}"
                    target_file_path = target_path / new_name
                    i += 1

                print(f"警告：文件 '{source_file.name}' 已存在。将重命名为 '{target_file_path.name}'")
                renamed_count += 1

            # 7. 执行拷贝操作
            try:
                # shutil.copy2 可以无缝接受 Path 对象
                shutil.copy2(source_file, target_file_path)
                # Path 对象在打印时会自动转换为适合当前操作系统的路径字符串
                print(f"已复制: {source_file} -> {target_file_path}")
                copied_count += 1
            except Exception as e:
                print(f"错误：复制文件 '{source_file}' 时失败: {e}")
                skipped_count += 1

    print("-" * 30)
    print("操作完成！")
    print(f"总计：成功复制 {copied_count} 个文件 (其中 {renamed_count} 个被重命名), 跳过 {skipped_count} 个。")


def batch_copy_files2dir(source_directory: str, target_directory: str):
    source_path: pathlib.Path = Path(source_directory)

    subdirs = [item for item in source_path.iterdir() if item.is_dir()]

    for subdir in subdirs:
        sub_name: str = subdir.name
        target_station_path = pathlib.Path(destination_directory) / sub_name
        copy_files_with_pathlib(str(subdir), str(target_station_path))
        pass


def batch_ws_files2dir(source: str, target_dir: str):
    """
        将 原始目录: source 下按照年月存储的 海洋站数据 => target_dir 统一存储
        存储形式为
            20240101_wd	 20240101_ws
                11	         4.2
                21	         4.2
                18	         3.7
                12	         3.8
                17	         4.3
                65	         1.7
                32	         2

    :param source:
    :param target_dir:
    :return:
    """
    source_path: pathlib.Path = Path(source)
    target_path: pathlib.Path = Path(target_dir)

    subdirs = [item for item in source_path.iterdir() if item.is_dir()]
    for subdir in subdirs:
        copied_count = 0
        skipped_count = 0
        renamed_count = 0
        sub_name: str = subdir.name
        target_station_path = pathlib.Path(destination_directory) / sub_name
        dict_station_df: dict = {}
        for source_file in subdir.rglob('WS????_DAT.*'):
            if source_file.is_file():
                # 批量读取文件，并按照站点名称存储至对应的 csv 中
                df = pd.read_csv(str(source_file), header=None, sep=r'\s+', engine='python')
                for index, row in df.iterrows():
                    if index > 0:
                        continue
                    # 第一列 (iloc[0]) 是 key，需要转为字符串
                    date_str = str(row.iloc[0])
                    date_key = date_str.split(".")[0]

                    # 剩余的列是 values
                    # .iloc[1:]: 选择从第二列到最后的所有数据
                    # .dropna(): 去除因行长度不同而产生的NaN值
                    # .values: 获取Numpy数组
                    values_array = row.iloc[1:].dropna().values
                    # eg: wd | ws
                    # 从第0个元素开始，每隔2个取一个，即 0, 2, 4, ...
                    wd_array = values_array[::2]
                    # 从第1个元素开始，每隔2个取一个，即 1, 3, 5, ...
                    ws_array = values_array[1::2]

                    key_wd = f'{date_key}_wd'
                    key_ws = f'{date_key}_ws'
                    dict_station_df[key_wd] = wd_array
                    dict_station_df[key_ws] = ws_array
                    pass
                # with open(str(source_file), 'r', encoding='utf-8') as f:
                #     # 同样是先一次性读入内存
                #     lines_in_memory = f.read().splitlines()
                #
                #     # 使用字典推导式处理内存中的列表
                #     # line.split() 只对非空行进行处理
                #
                #     dict_station_df = {
                #         parts[0]: np.array([float(v) for v in parts[1:]])
                #         for line in lines_in_memory if line.strip() and (parts := line.split())
                #     }
            # print(f"读取并写入db: {source_file}~")
        copied_count += 1
        save_path: pathlib.Path = target_path / f'{sub_name}.csv'
        df_temp: pd.DataFrame = pd.DataFrame(dict_station_df)
        df_temp.to_csv(str(save_path), index=False, encoding='utf-8-sig')
        print(f'当前站点存储完成:{str(save_path)}!')
    pass


def convert_local2utc_wind(input_path: pathlib.Path, output_path: pathlib.Path):
    """
        将 提取后的 站点原始数据 local time => utc time
    :param input_path:
    :param output_path:
    :return:
    """
    try:
        # 读取原始数据
        df_source: pd.DataFrame = pd.read_csv(str(input_path))
    except FileNotFoundError:
        print(f'错误: {input_path}文件不存在!')
        return
    # 获取所有列名并取出 columns 对应的 日期字符串
    all_columns = df_source.columns
    # 唯一的时间集合
    unique_dates = sorted(list(set([col.split('_')[0] for col in all_columns])))

    # 创建一个新的DataFrame来存储转换后的UTC数据
    df_utc = pd.DataFrame()
    #
    print(f'共计处理了:{len(unique_dates)}个时次的实况')
    for index in range(len(unique_dates) - 1):
        # 当前时间以及下一个时次的 date str
        temp_current_date_str: str = unique_dates[index]
        temp_next_date_str: str = unique_dates[index + 1]

        # 当前时次 以及 下一个时次 的 columns name
        current_wd_col = f"{temp_current_date_str}_wd"
        current_ws_col = f"{temp_current_date_str}_ws"
        next_wd_col = f"{temp_next_date_str}_wd"
        next_ws_col = f"{temp_next_date_str}_ws"

        # 必须存在的列
        required_cols = [current_wd_col, current_ws_col, next_wd_col, next_ws_col]
        if not all(col in df_source.columns for col in required_cols):
            print(f"警告：跳过日期 {temp_current_date_str}，因为它或其后一天的数据列不完整。")
            continue

        # --- 处理风向 (wd) ---
        # 1. 取本地时当天数据的后13个值 (对应UTC 00:00 - 12:00)
        # 本地时 08:00-20:00 的数据，在24行数据中是第12行到第24行 (索引为11到23)
        wd_part1 = df_source[current_wd_col].iloc[11:].reset_index(drop=True)
        # 2. 取本地时后一天数据的前11个值 (对应UTC 13:00 - 23:00)
        # 本地时 21:00-07:00 的数据，在24行数据中是第1行到第11行 (索引为0到10)
        wd_part2 = df_source[next_wd_col].iloc[:11].reset_index(drop=True)
        # 3. 拼接成一个完整的UTC日
        utc_wd_series = pd.concat([wd_part1, wd_part2], ignore_index=True)

        # --- 处理风速 (ws) ---
        # 重复风向的处理过程
        ws_part1 = df_source[current_ws_col].iloc[11:].reset_index(drop=True)
        ws_part2 = df_source[next_ws_col].iloc[:11].reset_index(drop=True)
        utc_ws_series = pd.concat([ws_part1, ws_part2], ignore_index=True)

        # 将生成的世界时数据添加到新的DataFrame中
        df_utc[f"{temp_current_date_str}_wd"] = utc_wd_series
        df_utc[f"{temp_current_date_str}_ws"] = utc_ws_series

        print(f"已成功转换 {temp_current_date_str} 的数据。")

        pass

    if not df_utc.empty:
        df_utc.to_csv(str(output_path), index=False, encoding='utf-8-sig')
        print(f'标准化处理完成:{str(output_path)}!')
    else:
        print('标准化处理失败!')


# --- 使用示例 ---
if __name__ == "__main__":
    # 请修改为您自己的路径
    # pathlib 可以很好地处理不同操作系统的路径分隔符
    source_directory = r"E:\01DATA\ML\渤海\渤海"  # <-- 需要拷贝的源文件夹
    destination_directory = r"E:\01DATA\ML\STATION_MERGE"  # <-- 所有文件最终存放的文件夹
    filter_dir = r'E:\01DATA\ML\STATION_FILTER_WS'
    save_dir = r'E:\01DATA\ML\STATION_WS'
    convert_dir = r'E:\01DATA\ML\STATION_CONVERT_WS'
    # step1: 将原始路径下的所有海洋站的实况数据按照海洋站名称存储至目标路径下
    # batch_copy_files2dir(source_directory, destination_directory)
    # step2: 读取目标路径下按照站名批量读取 WS0303_DAT.01111 文件
    # batch_ws_files2dir(filter_dir, save_dir)
    # step3: 将 local time [-21,+20] 对应的[wd,ws] => utc time [0,23]
    input_station_path = pathlib.Path(save_dir) / '鲅鱼圈.csv'
    output_station_path = pathlib.Path(save_dir) / '鲅鱼圈_stand.csv'
    for file in pathlib.Path(save_dir).rglob('*.csv'):
        source_file_name = file.name
        convert_file_name = source_file_name.replace('.csv', '_standard.csv')
        source_path = file
        copy_file = pathlib.Path(convert_dir) / convert_file_name
        convert_local2utc_wind(source_path, copy_file)
    pass
