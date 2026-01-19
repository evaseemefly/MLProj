import pandas as pd
import numpy as np
import re


# ==========================================
# 辅助函数：处理度分秒坐标字符串
# ==========================================
def dms_to_decimal(dms_str):
    """
    将 "120°35.71′E" 或 "39°30.02′N" 转换为浮点数坐标
    """
    if pd.isna(dms_str):
        return np.nan

    # 使用正则表达式提取数值
    # 匹配: 数字(度) + ° + 数字(分) + ′ + 方向
    match = re.match(r"(\d+)°([\d.]+)′([NSEW])", str(dms_str))
    if not match:
        return np.nan

    degrees = float(match.group(1))
    minutes = float(match.group(2))
    direction = match.group(3)

    decimal = degrees + minutes / 60.0

    # 如果是南纬(S)或西经(W)，设为负数
    if direction in ['S', 'W']:
        decimal = -decimal

    return decimal


# ==========================================
# 辅助函数：风速风向转 U/V 分量
# ==========================================
def wind_to_uv(ws, wd):
    """
    气象定义：0度是北风（从北吹向南），90度是东风。
    U: 纬向风 (西->东为正)
    V: 经向风 (南->北为正)
    """
    # 将角度转为弧度
    rad = np.radians(wd)

    # 气象公式
    u = -ws * np.sin(rad)
    v = -ws * np.cos(rad)
    return u, v


def main():
    # ==========================================
    # 主处理逻辑
    # ==========================================

    # 假设文件路径字典
    buoy_files = {
        "MF01004": "/Volumes/WD_BLACK/DATA/EXPORT/MF01004_2024_data_UTC.csv",
        "MF01002": "/Volumes/WD_BLACK/DATA/EXPORT/MF01002_2024_data_UTC.csv",
        "MF01001": "/Volumes/WD_BLACK/DATA/EXPORT/MF01001_2024_data_UTC.csv",
    }

    cleaned_dfs = []

    for buoy_id, file_path in buoy_files.items():
        print(f"/n正在处理浮标: {buoy_id} ...")

        # 1. 读取文件 (根据您的样例，看起来像CSV或Excel)
        # 假设是CSV，如果是Excel请用 pd.read_excel
        df = pd.read_csv(file_path)

        # 2. 增加 Buoy_ID 列 (关键!)
        df['Buoy_ID'] = buoy_id

        # 3. 时间格式转换
        # 您的格式: 202312311600 -> datetime
        #          202312311600
        df['Obs_Time'] = pd.to_datetime(df['TimeIndex'].astype(str), format='%Y%m%d%H%M')

        # 4. 坐标转换 (字符串 -> 浮点)
        df['Lon_Decimal'] = df['Longitude'].apply(dms_to_decimal)
        df['Lat_Decimal'] = df['Latitude'].apply(dms_to_decimal)

        # 5. 计算 U/V 分量
        # 确保 WS 和 WD 是数值型
        df['WS'] = pd.to_numeric(df['WS'], errors='coerce')
        df['WD'] = pd.to_numeric(df['WD'], errors='coerce')

        df['Obs_U'], df['Obs_V'] = wind_to_uv(df['WS'], df['WD'])

        # 6. (可选) 筛选有效列，给列重命名以符合后续流程
        # 只保留我们需要的列，减小体积
        keep_cols = ['Buoy_ID', 'Obs_Time', 'Lat_Decimal', 'Lon_Decimal', 'Obs_U', 'Obs_V', 'WS', 'WD']

        # 注意：您的原始数据里可能有 Nan，建议在这里简单丢弃关键数据缺失的行
        df_clean = df[keep_cols].dropna(subset=['Obs_U', 'Obs_V', 'Lat_Decimal'])

        # 重命名列以匹配我们之前的代码逻辑
        df_clean = df_clean.rename(columns={
            'Lat_Decimal': 'Lat',  # 这里只是参考，后续还会被 Target_Lat 覆盖
            'Lon_Decimal': 'Lon'
        })

        cleaned_dfs.append(df_clean)

    # ==========================================
    # 合并保存
    # ==========================================
    final_obs_df = pd.concat(cleaned_dfs, ignore_index=True)

    # 排序 (按时间，这会加速后续处理)
    final_obs_df = final_obs_df.sort_values(by='Obs_Time')

    # 保存为“清洗后的观测大表”
    final_obs_df.to_csv("buoy_obs_all_cleaned.csv", index=False)

    print("处理完成！")
    print(final_obs_df.head())


if __name__ == '__main__':
    main()
