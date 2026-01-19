import pandas as pd
import numpy as np
import xarray as xr
import os
import re

# ================= 配置区域 =================
# 1. 输入文件路径 (浮标实况数据)
OBS_FILE_PATH = '/Volumes/WD_BLACK/DATA/CONVERTED/MF01001_2024_data_UTC.csv'

# 2. 输出文件路径 (最终训练集)
OUTPUT_FILE = 'MF01001_2024_data_UTC_SPLITED.csv'

# 3. GRAPES 数据根目录 (请根据实际情况修改)
GRAPES_DIR = '/Volumes/WD_BLACK/DATA/CONVERTED/'

# 4. MF01001 的分段配置 (严格按照之前的分析结果)
# 格式: (结束时间, 纬度, 经度)
PHASE_CONFIG = [
    # Phase 1: 年初 -> 9月5日 00:00
    {'end': '2024-09-05 00:00', 'lat': 39.5005, 'lon': 120.5951},

    # Phase 2: 9月5日 01:00 -> 10月18日 23:00
    {'end': '2024-10-18 23:00', 'lat': 39.5051, 'lon': 120.6099},

    # Phase 3: 10月19日 00:00 -> 年底 (设一个足够远的时间)
    {'end': '2025-12-31 23:59', 'lat': 39.5031, 'lon': 120.6094}
]


# ================= 工具函数 =================

def dms_to_decimal(dms_str):
    """解析经纬度"""
    if pd.isna(dms_str): return np.nan
    dms_str = str(dms_str).strip()
    if dms_str in ['0', '0.0', '', 'nan']: return np.nan
    match = re.match(r"(\d+)°([\d.]+)[′']\s*([NSEW])?", dms_str)
    if match:
        deg = float(match.group(1))
        min_v = float(match.group(2))
        direction = match.group(3)
        val = deg + min_v / 60.0
        if direction and direction in ['S', 'W']: val = -val
        return val
    return np.nan


def get_current_target_coords(current_time):
    """根据当前时间，返回应该使用的中心坐标"""
    for phase in PHASE_CONFIG:
        end_time = pd.Timestamp(phase['end'])
        if current_time <= end_time:
            return phase['lat'], phase['lon']
    return PHASE_CONFIG[-1]['lat'], PHASE_CONFIG[-1]['lon']


def get_grapes_data_simulation(target_time, target_lat, target_lon):
    """
    【核心接口】从 GRAPES 获取数据
    注意：这里是一个模拟函数。您需要将其替换为真实的 xarray 读取代码。
    """
    # -----------------------------------------------------------
    # TODO: 请在此处替换为您真实的读取逻辑
    # 示例逻辑:
    # 1. 根据 target_time 找到对应的 .nc 文件路径
    #    filename = f"GRAPES_{target_time.strftime('%Y%m%d%H')}.nc"
    #    filepath = os.path.join(GRAPES_DIR, filename)
    #
    # 2. 使用 xarray 打开并插值
    #    if os.path.exists(filepath):
    #        ds = xr.open_dataset(filepath)
    #        # 提取风速 (假设变量名为 u10, v10)
    #        val = ds.interp(lat=target_lat, lon=target_lon, method='linear')
    #        u = val['u10'].values
    #        v = val['v10'].values
    #        ws = np.sqrt(u**2 + v**2)
    #        wd = (270 - np.degrees(np.arctan2(v, u))) % 360
    #        return ws, wd
    #    else:
    #        return np.nan, np.nan
    # -----------------------------------------------------------

    # # --- 模拟数据 (仅用于测试代码跑通) ---
    # # 模拟一个带随机噪声的风速，说明坐标在变
    # mock_ws = 5.0 + np.random.normal(0, 1) + (target_lat - 39.5) * 10
    # mock_wd = 180.0
    # return mock_ws, mock_wd


# ================= 主程序 =================

def main():
    print("1. 正在读取并清洗浮标实况数据...")
    df = pd.read_csv(OBS_FILE_PATH, index_col='TimeIndex')
    df.index = pd.to_datetime(df.index, format='%Y%m%d%H%M')

    # 强制重索引为 2024 全年 (UTC)
    full_range = pd.date_range(start='2023-12-31 16:00', end='2024-12-31 15:00', freq='H')
    df_full = df.reindex(full_range)
    df_full.index.name = 'Time'

    # 提取实况的风速风向 (假设 CSV 列名为 WS, WD)
    # 如果您的列名不同，请修改这里
    obs_ws_col = 'WS'
    obs_wd_col = 'WD'

    # 准备结果容器
    result_data = []

    print(f"2. 开始构建数据集 (共 {len(df_full)} 个时间步)...")
    print("   正在进行【分段时空拼接】提取...")

    # 遍历每一行
    for current_time, row in df_full.iterrows():

        # A. 获取浮标实况 (Label)
        # 即使浮标漂移了，记录的实况也是真实的
        real_obs_ws = row.get(obs_ws_col, np.nan)
        real_obs_wd = row.get(obs_wd_col, np.nan)

        # B. 动态决定提取坐标 (Input Feature)
        target_lat, target_lon = get_current_target_coords(current_time)

        # C. 从 GRAPES 提取 (需要您完善 get_grapes_data_simulation 函数)
        model_ws, model_wd = get_grapes_data_simulation(current_time, target_lat, target_lon)

        # D. 构造数据行
        result_data.append({
            'Time': current_time,
            'Buoy_ID': 'MF01001',

            # 关键：将当前使用的提取坐标也作为特征写入
            # 这样模型能知道“这个数据是来自哪个位置的”
            'Feature_Lat': target_lat,
            'Feature_Lon': target_lon,

            # 模型预报 (Input)
            'Model_WS': model_ws,
            'Model_WD': model_wd,

            # 浮标实况 (Label/Target)
            'Obs_WS': real_obs_ws,
            'Obs_WD': real_obs_wd
        })

        # 打印进度 (每1000条)
        if len(result_data) % 1000 == 0:
            print(f"   已处理: {current_time} | 当前锁定坐标: ({target_lat:.4f}, {target_lon:.4f})")

    # 转换回 DataFrame
    df_result = pd.DataFrame(result_data)
    df_result.set_index('Time', inplace=True)

    # 3. 保存结果
    df_result.to_csv(OUTPUT_FILE)
    print("=" * 60)
    print(f"训练集构建完成！已保存至: {OUTPUT_FILE}")
    print(f"数据形状: {df_result.shape}")
    print("前5行预览:")
    print(df_result[['Feature_Lat', 'Model_WS', 'Obs_WS']].head())
    print("\n检查跳变点 (9月5日) 附近的数据:")
    print(df_result.loc['2024-09-04 22:00':'2024-09-05 02:00', ['Feature_Lat', 'Feature_Lon']])


if __name__ == "__main__":
    main()