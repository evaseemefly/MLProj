

import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# ================= 配置区域 =================
FILE_PATH = '/Volumes/WD_BLACK/DATA/CONVERTED/MF01004_2024_data_UTC.csv'


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

    time_colors = np.linspace(0, 1, len(df_plot))

    # === 【关键修改】添加抖动 (Jitter) ===
    # 计算抖动幅度：取数据精度的 1/4 左右
    # 原始精度约 0.01分 = 0.00016度，我们抖动 0.00005度
    jitter_amount = 0.00005

    # 生成随机噪声
    lat_jitter = np.random.normal(0, jitter_amount, size=len(df_plot))
    lon_jitter = np.random.normal(0, jitter_amount, size=len(df_plot))

    # 绘图时加上噪声 (仅用于绘图，不修改原始数据)
    sc = plt.scatter(df_plot['Lon_Decimal'] + lon_jitter,
                     df_plot['Lat_Decimal'] + lat_jitter,
                     c=time_colors, cmap='viridis',
                     s=20, alpha=0.3, label='Daily Position (Jittered)')
    # ===================================

    plt.scatter(final_mean_lon, final_mean_lat, c='red', marker='*', s=300,
                label='Mean Position', zorder=10, edgecolors='black')

    # 5. 动态设置坐标轴范围 (让图表只显示浮标周围)
    # 在数据范围基础上再向外扩一点点 (0.005度 ≈ 500米) 留白
    margin = 0.005
    plt.xlim(df_plot['Lon_Decimal'].min() - margin, df_plot['Lon_Decimal'].max() + margin)
    plt.ylim(df_plot['Lat_Decimal'].min() - margin, df_plot['Lat_Decimal'].max() + margin)

    # 强制不使用科学计数法 (防止坐标轴显示不直观)
    plt.ticklabel_format(useOffset=False, style='plain')

    plt.title(f'Buoy Watch Circle (Filtered)\nValid Data: {len(df_plot)} points', fontsize=14)
    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.colorbar(sc, label='Time (Start -> End)')

    # 保持比例尺一致 (非常重要，否则圆会变成椭圆)
    plt.axis('equal')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()