import pathlib
from typing import List

from arrow import arrow
# TODO:[-] 25-08-05 注意使用lxml库，不使用xml库，lxml可以更好的处理encoder的问题
# import xml.etree.ElementTree as ET
from lxml import etree as ET
import pandas as pd
import numpy as np
from tqdm import tqdm  # 引入tqdm库，用于显示进度条，让等待过程更直观

fub_codes = ['MF01001', 'MF01002', 'MF01004', 'MF02004', 'MF02001']
"""浮标code集合"""


def process_buoy_data_for_year(dir_path: str, target_year: int) -> pd.DataFrame:
    """
    遍历指定目录，读取特定年份的浮标数据XML文件，并将其转换为Pandas DataFrame。

    Args:
        directory_path (str): 存放XML文件的数据目录。
        target_year (int): 需要处理的目标年份，例如 2024。

    Returns:
        pd.DataFrame: 包含所有提取数据的DataFrame。
    """
    all_data = []  # 用于存储从所有文件中提取的数据
    # 将输入路径统一转换为 Path 对象，这是 pathlib 的标准实践
    data_dir = pathlib.Path(dir_path)

    # 检查目录是否存在
    if not data_dir.is_dir():
        print(f"错误：目录 '{data_dir}' 不存在。")
        return pd.DataFrame()

    # 使用 Path.glob() 方法直接筛选出目标文件，返回一个生成器
    # 模式 f"{target_year}*.xml" 表示以年份开头、以 .xml 结尾的所有文件
    files_to_process = list(data_dir.glob(f"{target_year}*.xml"))

    if not files_to_process:
        print(f"警告：在目录 '{dir_path}' 中未找到 {target_year} 年的XML文件。")
        return pd.DataFrame()

    print(f"开始处理 {target_year} 年的数据，共找到 {len(files_to_process)} 个文件...")

    # 创建一个可以处理 GB2312 编码的 XML 解析器
    # parser = ET.XMLParser(encoding="gb2312")

    # 使用tqdm创建进度条
    for file_path in tqdm(files_to_process, desc=f"处理 {target_year} 年文件"):
        try:
            # 解析XML文件
            tree = ET.parse(str(file_path))
            # tree = ET.parse(file_path, parser=parser)
            root = tree.getroot()

            # --- 数据提取 ---
            # 使用 .// 可以在整个树中查找节点，更具鲁棒性
            location_node = root.find('.//Location')
            datetime_node = root.find('.//DateTime')
            buoydata_node = root.find('.//BuoyData')

            # 确保节点存在，避免因文件结构差异而出错
            if location_node is not None and datetime_node is not None and buoydata_node is not None:
                data_row = {
                    'DT': datetime_node.get('DT'),
                    'longitude': location_node.get('longitude'),
                    'latitude': location_node.get('latitude'),
                    'WS': buoydata_node.get('WS'),
                    'WD': buoydata_node.get('WD'),
                    'WSE': buoydata_node.get('WSE'),
                    'WSM': buoydata_node.get('WSM'),
                }
                all_data.append(data_row)
            else:
                # 使用 file_path.name 获取文件名，比完整路径更清晰
                print(f"\n警告：文件 '{file_path.name}' 结构不完整，已跳过。")


        except ET.ParseError:
            print(f"\n错误：文件 '{file_path.name}' XML格式错误，无法解析。")
        except Exception as e:
            print(f"\n处理文件 '{file_path.name}' 时发生未知错误: {e}")

    if not all_data:
        print("处理完成，但没有成功提取到任何数据。")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    print("\n数据处理完成！")
    return df


def main():
    code: str = 'MF02004'
    dir_path: str = r'/Volumes/DRCC_DATA/01DATA/02FUB/历史数据/MF02004/2024/all'
    out_put_path: str = r'/Volumes/DRCC_DATA/01DATA/02FUB/历史数据/处理后'
    year = 2024
    data_dir_path = pathlib.Path(dir_path)
    buoy_df = process_buoy_data_for_year(data_dir_path, year)
    # 3. 显示结果 (逻辑与之前相同)
    if not buoy_df.empty:
        print("\n成功生成的DataFrame信息如下：")
        buoy_df.info()

        print("\nDataFrame内容预览：")
        print(buoy_df)

        # 4保存到CSV文件
        output_csv_path = str(pathlib.Path(out_put_path) / f"buoy_data_{code}_{year}.csv")
        buoy_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n数据已保存到文件: {output_csv_path}")


if __name__ == '__main__':
    main()
    pass
