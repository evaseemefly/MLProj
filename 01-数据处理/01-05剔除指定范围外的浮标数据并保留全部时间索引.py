import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# ================= 配置区域 =================
FILE_PATH = '/Volumes/WD_BLACK/DATA/CONVERTED/MF01001_2024_data_UTC.csv'


# ===========================================

def dms_to_decimal(dms_str):
    """
    将度分格式字符串转换为小数度数
    支持格式:
    1. 120°35.70′E  (带方位)
    2. 120°36.60′   (不带方位, 默认为正)
    """
    if pd.isna(dms_str): return np.nan
    dms_str = str(dms_str).strip()

    # 过滤无效值
    if dms_str in ['0', '0.0', '', 'nan']: return np.nan

    # === 正则表达式修改点 ===
    # 1. ([NSEW])? : 这里的 ? 表示前面的方位字母分组是“可选的”（0次或1次匹配）
    # 2. \s* : 允许分符号和方位字母之间有空格
    # 3. [′'] : 兼容中文全角撇号(′)和英文单引号(')
    match = re.match(r"(\d+)°([\d.]+)[′']\s*([NSEW])?", dms_str)

    if match:
        deg = float(match.group(1))
        min_v = float(match.group(2))
        direction = match.group(3)  # 如果没有方位，这里会是 None

        val = deg + min_v / 60.0

        # 只有明确指定了 S (南纬) 或 W (西经) 时才取负
        if direction and direction in ['S', 'W']:
            val = -val

        return val

    return np.nan


def main():
    # 1. 读取与转换
    df = pd.read_csv(FILE_PATH, index_col='TimeIndex', dtype={'TimeIndex': str})
    df['Lat_Decimal'] = df['Latitude'].apply(dms_to_decimal)
    df['Lon_Decimal'] = df['Longitude'].apply(dms_to_decimal)

    # 初步清洗无效值
    df_valid_ = df.dropna(subset=['Lat_Decimal', 'Lon_Decimal'])

    if len(df_valid_) == 0:
        print("没有有效数据")
        return

    # 2. 【关键步骤】计算中心并剔除离群点
    # 先计算一个粗略的中位数或者均值
    center_lat = df_valid_['Lat_Decimal'].median()
    center_lon = df_valid_['Lon_Decimal'].median()

    print(f"初步中心位置: {center_lon:.4f}, {center_lat:.4f}")

    # 设定保留范围：只保留中心点附近 +/- 0.1 度的数据 (约 +/- 10公里)
    # 对于锚系浮标，0.1度已经非常大了，足够包含旋回范围
    LIMIT_DEGREE = 0.03

    # 创建掩码 Mask：标记哪些是“正常点”
    is_normal_mask = (
            df['Lat_Decimal'].notna() &
            df['Lon_Decimal'].notna() &
            (np.abs(df['Lat_Decimal'] - center_lat) < LIMIT_DEGREE) &
            (np.abs(df['Lon_Decimal'] - center_lon) < LIMIT_DEGREE)
    )

    outliers_count = len(df) - is_normal_mask.sum()
    print(f"正常点数量: {is_normal_mask.sum()}")
    print(f"异常/缺失点数量: {outliers_count}")

    # 4. 计算固定平均坐标 (只用正常点算)
    final_mean_lat = df.loc[is_normal_mask, 'Lat_Decimal'].mean()
    final_mean_lon = df.loc[is_normal_mask, 'Lon_Decimal'].mean()

    print(f"\n>>> 最终选定的固定坐标 (用于GRAPES提取):")
    print(f">>> Lat: {final_mean_lat:.6f}")
    print(f">>> Lon: {final_mean_lon:.6f}")
    pass



if __name__ == "__main__":
    main()
