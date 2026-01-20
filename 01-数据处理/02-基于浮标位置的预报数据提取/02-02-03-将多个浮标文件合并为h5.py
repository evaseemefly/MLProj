import pandas as pd
import numpy as np
import re


# ==========================================
# 辅助函数：处理度分秒坐标字符串
# ==========================================
def dms_to_decimal(dms_str):
    """
    TODO:[-] 26-01-20 兼容两种经纬度格式
    兼容转换两种格式：
    1. "120°35.71′E" (带方向)
    2. "120°36.58′"  (不带方向，默认视为N/E)
    """
    if pd.isna(dms_str):
        return np.nan

    # 转化为字符串并去除首尾空格
    s = str(dms_str).strip()

    # 1. 修改正则表达式
    # (\d+)       : 匹配度
    # °           : 匹配度符号
    # ([\d.]+)    : 匹配分 (支持整数或小数)
    # ′           : 匹配分符号
    # ([NSEW])?   : 关键修改！添加 '?' 表示方向后缀是可选的 (出现0次或1次)
    pattern = r"(\d+)°([\d.]+)′([NSEW])?"

    match = re.match(pattern, s)
    if not match:
        # 如果匹配失败，可以尝试打印一下出错的数据以便调试
        # print(f"无法解析的格式: {s}")
        return np.nan

    degrees = float(match.group(1))
    minutes = float(match.group(2))
    direction = match.group(3)  # 如果没有方向，这里会是 None

    # 计算十进制坐标
    decimal = degrees + minutes / 60.0

    # 2. 处理方向逻辑
    # 只有当明确出现了 'S' (南纬) 或 'W' (西经) 时才取负
    # 如果 direction 是 None (格式2)，或者 'N'/'E'，都保持正数
    if direction and direction in ['S', 'W']:
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
    h5_path: str = r"/Volumes/WD_BLACK/DATA/EXPORT/buoy_obs_all_cleaned_260120_02.h5"

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
        # df_clean = df[keep_cols].dropna(subset=['Obs_U', 'Obs_V', 'Lat_Decimal'])
        df_clean = df[keep_cols]
        cleaned_dfs.append(df_clean)

    # ==========================================
    # 合并保存
    # ==========================================
    with pd.HDFStore(h5_path, mode='w', complib='blosc', complevel=9) as store:
        # 获取所有唯一的浮标 Code

        for temp_df in cleaned_dfs:
            buoy_id = temp_df['Buoy_ID'][0]
            print(f"正在存储浮标: {buoy_id} ...")

            # 筛选出该浮标的数据
            df_sub = temp_df

            # 【优化建议】：存入前把时间设为索引并排序
            # 这样读出来直接就是时间序列，非常方便后续插值
            df_sub = df_sub.set_index('Obs_Time').sort_index()

            # 存入 H5，Key 就是浮标 ID
            # format='table' 支持后续在不读取文件的情况下进行查询，但在我们这种拆分 key 的场景下，
            # format='fixed' (默认) 读写速度更快。这里用 fixed 即可。
            store.put(key=buoy_id, value=df_sub, format='fixed')

    print("H5 文件创建完成！")


if __name__ == '__main__':
    main()
