

import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# ================= 配置区域 =================
FILE_PATH = '/Volumes/WD_BLACK/DATA/CONVERTED/MF01001_2024_data_UTC.csv'


# ===========================================

def dms_to_decimal(dms_str):
    if pd.isna(dms_str): return np.nan
    dms_str = str(dms_str).strip()
    # 增加对 0 或 '0' 的过滤，防止解析出 (0,0)
    if dms_str == '0' or dms_str == '0.0': return np.nan

    match = re.match(r"(\d+)°([\d.]+)′([NSEW])", dms_str)
    if match:
        deg = float(match.group(1))
        min_v = float(match.group(2))
        direction = match.group(3)
        val = deg + min_v / 60.0
        if direction in ['S', 'W']: val = -val
        return val
    return np.nan


def main():
    # 1. 读取与转换
    df = pd.read_csv(FILE_PATH, index_col='TimeIndex', dtype={'TimeIndex': str})
    df['Lat_Decimal'] = df['Latitude'].apply(dms_to_decimal)
    df['Lon_Decimal'] = df['Longitude'].apply(dms_to_decimal)

    # 初步清洗无效值
    df_clean = df.dropna(subset=['Lat_Decimal', 'Lon_Decimal'])

    if len(df_clean) == 0:
        print("没有有效数据")
        return

    # 2. 【关键步骤】计算中心并剔除离群点
    # 先计算一个粗略的中位数或者均值
    center_lat = df_clean['Lat_Decimal'].median()
    center_lon = df_clean['Lon_Decimal'].median()

    print(f"初步中心位置: {center_lon:.4f}, {center_lat:.4f}")

    # 设定保留范围：只保留中心点附近 +/- 0.1 度的数据 (约 +/- 10公里)
    # 对于锚系浮标，0.1度已经非常大了，足够包含旋回范围
    LIMIT_DEGREE = 0.03

    condition = (
            (np.abs(df_clean['Lat_Decimal'] - center_lat) < LIMIT_DEGREE) &
            (np.abs(df_clean['Lon_Decimal'] - center_lon) < LIMIT_DEGREE)
    )

    # 获取过滤后的数据用于绘图
    df_plot = df_clean[condition]
    outliers_count = len(df_clean) - len(df_plot)
    print(f"剔除了 {outliers_count} 个异常漂移/错误点 (如 0,0 坐标)")

    # 剔除后的df shape 为：
    # (5926, 120)
    # 共5926个时刻的轨迹
    # 3. 重新计算精确的平均位置 (基于过滤后的数据)
    final_mean_lat = df_plot['Lat_Decimal'].mean()
    final_mean_lon = df_plot['Lon_Decimal'].mean()

    # 4. 绘图
    plt.figure(figsize=(10, 8), dpi=120)

    # === 【修改点】使用 hexbin (六边形分箱图) ===
    # gridsize: 控制格子的密度。数值越大，格子越小，越精细。建议设为 40-60。
    # cmap='YlOrRd': 颜色映射，Yellow -> Orange -> Red (由浅黄到深红)。
    # mincnt=1: 计数为0的格子不显示颜色（透明/白色）。
    hb = plt.hexbin(df_plot['Lon_Decimal'], df_plot['Lat_Decimal'],
                    gridsize=50, cmap='YlOrRd', mincnt=1,
                    edgecolors='none')  # edgecolors='none' 去掉格子边框，看起来更像连续热图

    # 添加颜色条，显示具体的点数
    cb = plt.colorbar(hb, label='Point Count (Density)')
    # ==========================================

    # 绘制平均位置（红色五角星）
    # 为了防止红色五角星混入背景的红色热图中，建议给五角星加个明显的黑边，或者换成青色/蓝色
    plt.scatter(final_mean_lon, final_mean_lat, c='cyan', marker='*', s=300,
                label='Mean Position', zorder=10, edgecolors='black', linewidth=1.5)

    # 5. 动态设置坐标轴范围
    margin = 0.005
    plt.xlim(df_plot['Lon_Decimal'].min() - margin, df_plot['Lon_Decimal'].max() + margin)
    plt.ylim(df_plot['Lat_Decimal'].min() - margin, df_plot['Lat_Decimal'].max() + margin)

    plt.ticklabel_format(useOffset=False, style='plain')

    plt.title(f'Buoy Density Heatmap (Hexbin)\nValid Data: {len(df_plot)} points', fontsize=14)
    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    plt.grid(True, linestyle='--', alpha=0.3)  # 网格线淡一点，不要干扰热图
    plt.legend()  # 显示五角星的图例

    plt.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()