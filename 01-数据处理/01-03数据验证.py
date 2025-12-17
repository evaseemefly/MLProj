import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as ET
import random

# ================= 配置区域 =================
# 根目录
BASE_DIR = '/Users/evaseemefly/03data/02fub'
SOURCE_DIR = os.path.join(BASE_DIR, 'source')
EXPORT_DIR = os.path.join(BASE_DIR, 'export')

# 要验证的文件
TARGET_CSV = 'MF01002_2024_data_UTC.csv'
BUOY_ID = 'MF01002'
YEAR = '2024'

# 抽样数量
SAMPLE_SIZE = 10


# ===========================================

def clean_value(val):
    """
    与提取脚本一致的数据清洗逻辑，用于处理XML原值
    """
    if val is None:
        return np.nan
    val = str(val).strip()
    if val in ['/', '', 'XXX.X', 'XXX', 'nan', 'NULL']:
        return np.nan
    try:
        return float(val)
    except ValueError:
        return val


def parse_xml_file(filepath):
    """
    重新解析XML文件以获取“真值”
    """
    data_dict = {}
    try:
        with open(filepath, 'r', encoding='gb18030', errors='replace') as f:
            xml_content = f.read()

        root = ET.fromstring(xml_content)
        rpt = root.find('BuoyageRpt')
        if rpt is None: return None

        # 1. BuoyInfo
        buoy_info = rpt.find('BuoyInfo')
        if buoy_info is not None:
            loc = buoy_info.find('Location')
            if loc is not None:
                data_dict['Longitude'] = clean_value(loc.get('longitude'))
                data_dict['Latitude'] = clean_value(loc.get('latitude'))

        # 2. HugeBuoyData
        huge_data = rpt.find('HugeBuoyData')
        if huge_data is not None:
            # RunningStatus
            run_status = huge_data.find('RunningStatus')
            if run_status is not None:
                for k, v in run_status.attrib.items():
                    data_dict[f'RunStatus_{k}'] = clean_value(v)
            # BuoyData
            buoy_data_tag = huge_data.find('BuoyData')
            if buoy_data_tag is not None:
                for k, v in buoy_data_tag.attrib.items():
                    data_dict[k] = clean_value(v)
            # SeaCurrent
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
                                data_dict[f"SCurrent_{k}_{layer_suffix}"] = clean_value(v)
            # TempSalt
            temp_salt = huge_data.find('TempSalt')
            if temp_salt is not None:
                ts_tag = temp_salt.find('TSalt')
                if ts_tag is not None:
                    for k, v in ts_tag.attrib.items():
                        data_dict[f'TSalt_{k}'] = clean_value(v)
    except Exception as e:
        print(f"  [Error] 无法解析原始文件: {e}")
        return None
    return data_dict


def compare_values(csv_val, xml_val, col_name):
    """
    比较两个值是否相等（处理浮点数和NaN）
    """
    # 1. 处理 NaN
    if pd.isna(csv_val) and pd.isna(xml_val):
        return True, "Match (NaN)"
    if pd.isna(csv_val) or pd.isna(xml_val):
        return False, f"Mismatch (One is NaN): CSV={csv_val}, XML={xml_val}"

    # 2. 处理数值
    try:
        c_float = float(csv_val)
        x_float = float(xml_val)
        if np.isclose(c_float, x_float, atol=1e-5):
            return True, f"Match ({c_float})"
        else:
            return False, f"Mismatch (Value): CSV={c_float}, XML={x_float}"
    except ValueError:
        # 3. 处理字符串
        if str(csv_val).strip() == str(xml_val).strip():
            return True, f"Match (String: {csv_val})"
        else:
            return False, f"Mismatch (String): CSV={csv_val}, XML={xml_val}"


def main():
    csv_path = os.path.join(EXPORT_DIR, TARGET_CSV)
    print(f"正在加载数据集: {csv_path} ...")

    # 读取CSV，确保索引是字符串以便解析
    df = pd.read_csv(csv_path, index_col='TimeIndex', dtype={'TimeIndex': str})

    # 过滤掉全空的行（可选，如果想验证空文件是否正确生成了NaN，可以不过滤）
    # 这里我们优先验证“有数据”的行
    valid_indices = df.dropna(how='all').index.tolist()

    if len(valid_indices) < SAMPLE_SIZE:
        print("警告: 有效数据行少于抽样数量，将验证所有有效行。")
        samples = valid_indices
    else:
        samples = random.sample(valid_indices, SAMPLE_SIZE)

    print(f"随机抽取了 {len(samples)} 个时间点进行验证...\n")
    print("=" * 60)

    for utc_idx_str in samples:
        # 1. 时间转换：CSV Index (UTC) -> File Name (BJT)
        # 格式: YYYYMMDDHHMM
        try:
            utc_dt = pd.to_datetime(utc_idx_str, format='%Y%m%d%H%M')
        except:
            print(f"[跳过] 索引格式错误: {utc_idx_str}")
            continue

        bjt_dt = utc_dt + pd.Timedelta(hours=8)

        # 2. 构建原始文件路径
        # 路径: /source/MF01001/2024/01/202401010800MF01001.dat.xml
        year_str = str(bjt_dt.year)
        month_str = bjt_dt.strftime('%m')
        file_name = f"{bjt_dt.strftime('%Y%m%d%H%M')}{BUOY_ID}.dat.xml"

        xml_path = os.path.join(SOURCE_DIR, BUOY_ID, year_str, month_str, file_name)

        print(f"\n验证时间点 (UTC): {utc_idx_str}")
        print(f"对应北京时 (BJT): {bjt_dt.strftime('%Y-%m-%d %H:%M')}")
        print(f"查找源文件: .../{BUOY_ID}/{year_str}/{month_str}/{file_name}")

        # 3. 获取 CSV 行数据
        csv_row = df.loc[utc_idx_str]

        # 4. 获取 XML 原始数据
        if not os.path.exists(xml_path):
            # 如果文件不存在，CSV 行应该全是 NaN (除了可能的时间戳列如果被保留的话)
            # 简单检查几个关键列
            if csv_row.notna().sum() > 0:
                print(f"❌ 错误: 源文件不存在，但 CSV 中有数据！")
            else:
                print(f"✅ 通过: 源文件不存在，CSV 为空行。")
            continue

        xml_data = parse_xml_file(xml_path)
        if xml_data is None:
            print("⚠️ 跳过: XML 解析失败")
            continue

        # 5. 逐个字段比对
        mismatch_count = 0
        match_count = 0

        # 遍历 XML 中提取到的所有字段
        for key, xml_val in xml_data.items():
            if key not in csv_row.index:
                print(f"  ⚠️ 警告: XML中有字段 {key}，但CSV中没有该列")
                continue

            csv_val = csv_row[key]
            is_match, msg = compare_values(csv_val, xml_val, key)

            if not is_match:
                print(f"  ❌ 不匹配 [{key}]: {msg}")
                mismatch_count += 1
            else:
                match_count += 1

        if mismatch_count == 0:
            print(f"✅ 完美匹配 (检查了 {match_count} 个字段)")
        else:
            print(f"❌ 存在 {mismatch_count} 个字段不匹配")

    print("\n" + "=" * 60)
    print("验证结束。")


if __name__ == "__main__":
    main()