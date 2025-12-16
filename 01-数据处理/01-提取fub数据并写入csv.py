import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ================= 配置区域 =================
# 浮标列表
FUBS = ['MF01001', 'MF01002']
# 数据的根目录 (父级)
BASE_ROOT = r'E:/01DATA/SOURCE/FUB'
YEAR = 2024


# ===========================================

def clean_value(val):
    """
    清洗数据：将 XML 中的 '/', 'XXX.X', 空字符串转换为 NaN
    """
    if val is None:
        return np.nan
    val = str(val).strip()
    # 增加一些常见的无效值标记
    if val in ['/', '', 'XXX.X', 'XXX', 'nan', 'NULL']:
        return np.nan
    return val


def parse_xml_file(filepath):
    """
    解析单个XML文件 (修正版：支持 GB2312/GBK 编码)
    """
    data_dict = {}
    try:
        # 【关键修改】：先用 gb18030 (兼容GB2312/GBK) 读取为字符串
        with open(filepath, 'r', encoding='gb18030', errors='replace') as f:
            xml_content = f.read()

        # 某些情况下，读取为 Unicode 字符串后，头部声明的 encoding="GB2312"
        # 可能会让解析器困惑，建议将其替换或移除，或者直接解析
        # 这里直接解析通常可行，如果还报错，可以取消下面这行的注释：
        # xml_content = xml_content.replace('encoding="GB2312"', '')

        root = ET.fromstring(xml_content)

        rpt = root.find('BuoyageRpt')
        if rpt is None: return None

        # 1. BuoyInfo (位置)
        buoy_info = rpt.find('BuoyInfo')
        if buoy_info is not None:
            loc = buoy_info.find('Location')
            if loc is not None:
                data_dict['Longitude'] = clean_value(loc.get('longitude'))
                data_dict['Latitude'] = clean_value(loc.get('latitude'))

        # 2. HugeBuoyData
        huge_data = rpt.find('HugeBuoyData')
        if huge_data is not None:
            # A. RunningStatus
            run_status = huge_data.find('RunningStatus')
            if run_status is not None:
                for k, v in run_status.attrib.items():
                    data_dict[f'RunStatus_{k}'] = clean_value(v)

            # B. BuoyData (气象要素)
            buoy_data_tag = huge_data.find('BuoyData')
            if buoy_data_tag is not None:
                for k, v in buoy_data_tag.attrib.items():
                    data_dict[k] = clean_value(v)

            # C. SeaCurrent (海流 - 多层展开)
            sea_current = huge_data.find('SeaCurrent')
            if sea_current is not None:
                for k, v in sea_current.attrib.items():
                    data_dict[f'SeaCurrent_Meta_{k}'] = clean_value(v)

                for current in sea_current.findall('SCurrent'):
                    layer_no = current.get('NO')
                    if layer_no:
                        layer_suffix = f"{int(layer_no):02d}"
                        for k, v in current.attrib.items():
                            if k != 'NO':
                                col_name = f"SCurrent_{k}_{layer_suffix}"
                                data_dict[col_name] = clean_value(v)

            # D. TempSalt (温盐)
            temp_salt = huge_data.find('TempSalt')
            if temp_salt is not None:
                ts_tag = temp_salt.find('TSalt')
                if ts_tag is not None:
                    for k, v in ts_tag.attrib.items():
                        data_dict[f'TSalt_{k}'] = clean_value(v)

    except Exception as e:
        # 打印具体的解析错误
        print(f"解析异常 [{filepath}]: {e}")
        return None

    return data_dict

def process_buoy(buoy_id, time_range):
    """
    处理单个浮标的全年数据
    """
    print(f"正在处理浮标: {buoy_id} ...")
    extracted_data = []

    for dt in time_range:
        # 构建路径逻辑:
        # 假设结构为: /root/MF01001/2024/01/文件名
        month_str = dt.strftime('%m')

        # 关键修改：路径中加入 buoy_id
        file_dir = os.path.join(BASE_ROOT, buoy_id, str(YEAR), month_str)

        # 文件名: 202401010000MF01001.dat.xml
        time_str = dt.strftime('%Y%m%d%H%M')
        filename = f"{time_str}{buoy_id}.dat.xml"
        filepath = Path(file_dir) / filename
        file_path_str: str = str(filepath)

        row_data = {'timestamp': dt}

        if os.path.exists(file_path_str):
            xml_data = parse_xml_file(file_path_str)
            if xml_data:
                row_data.update(xml_data)

        extracted_data.append(row_data)

    # 转DataFrame
    df = pd.DataFrame(extracted_data)
    df.set_index('timestamp', inplace=True)

    # 强制重索引确保时间连续（缺省填充NaN）
    df = df.reindex(time_range)

    # 类型转换
    df = df.apply(pd.to_numeric, errors='ignore')

    # 格式化索引
    df.index.name = 'TimeIndex'
    df.index = df.index.strftime('%Y%m%d%H%M')

    # 保存文件
    output_filename = f"{buoy_id}_{YEAR}_data.csv"
    df.to_csv(output_filename, encoding='utf-8-sig')
    print(f"  -> {buoy_id} 完成，保存为: {output_filename} (行数: {len(df)})")


def main():
    # 生成时间轴
    full_time_range = pd.date_range(start=f'{YEAR}-01-01 00:00', end=f'{YEAR}-12-31 23:00', freq='H')

    # 遍历浮标列表
    for buoy in FUBS:
        process_buoy(buoy, full_time_range)

    print("\n所有浮标处理完毕。")


if __name__ == "__main__":
    main()
