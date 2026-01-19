import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.cluster import KMeans

# ================= 配置区域 =================
# 请确保路径正确
FILE_PATH = '/Volumes/WD_BLACK/DATA/CONVERTED/MF01004_2024_data_UTC.csv'
CLUSTERS_NUM = 1  # 您识别出的簇数量（三团）
LIMIT_DEGREE = 0.03  # 您的空间过滤阈值


# ===========================================

def dms_to_decimal(dms_str):
    """
    将度分格式字符串转换为小数度数 (优化版正则)
    """
    if pd.isna(dms_str): return np.nan
    dms_str = str(dms_str).strip()
    if dms_str in ['0', '0.0', '', 'nan']: return np.nan

    # 兼容中文撇号、英文单引号、可选方位字母、空格
    match = re.match(r"(\d+)°([\d.]+)[′']\s*([NSEW])?", dms_str)

    if match:
        deg = float(match.group(1))
        min_v = float(match.group(2))
        direction = match.group(3)

        val = deg + min_v / 60.0
        if direction and direction in ['S', 'W']:
            val = -val
        return val
    return np.nan


def main():
    print("1. 正在读取数据...")
    df = pd.read_csv(FILE_PATH, index_col='TimeIndex')
    # 转换为 datetime 对象以便后续按时间排序
    df.index = pd.to_datetime(df.index, format='%Y%m%d%H%M')

    # 2. 坐标转换
    df['Lat_Decimal'] = df['Latitude'].apply(dms_to_decimal)
    df['Lon_Decimal'] = df['Longitude'].apply(dms_to_decimal)

    # 初步剔除空值
    df_clean = df.dropna(subset=['Lat_Decimal', 'Lon_Decimal'])

    if len(df_clean) == 0:
        print("没有有效数据")
        return

    # ========================================================
    # 【步骤 A】空间过滤 (剔除极端离群点)
    # ========================================================
    center_lat = df_clean['Lat_Decimal'].median()
    center_lon = df_clean['Lon_Decimal'].median()

    print(f"初步中心位置 (Median): {center_lon:.4f}, {center_lat:.4f}")

    # 应用 LIMIT_DEGREE = 0.03 过滤
    condition = (
            (np.abs(df_clean['Lat_Decimal'] - center_lat) < LIMIT_DEGREE) &
            (np.abs(df_clean['Lon_Decimal'] - center_lon) < LIMIT_DEGREE)
    )

    # 得到过滤后的有效数据 (用于后续聚类)
    df_valid = df_clean[condition].copy()

    outliers_count = len(df_clean) - len(df_valid)
    print(f"已剔除 {outliers_count} 个极端漂移点 (阈值 +/- {LIMIT_DEGREE}°)")
    print(f"剩余有效点数: {len(df_valid)}")

    # ========================================================
    # 【步骤 B】K-Means 聚类 (识别三团散点)
    # ========================================================
    print(f"正在执行 K-Means 聚类 (K={CLUSTERS_NUM})...")

    # 构造特征矩阵
    X = df_valid[['Lat_Decimal', 'Lon_Decimal']].values

    # 聚类
    kmeans = KMeans(n_clusters=CLUSTERS_NUM, random_state=42, n_init=10)
    df_valid['Cluster_Label'] = kmeans.fit_predict(X)

    # ========================================================
    # 【步骤 C】分析结果与分段统计
    # ========================================================
    cluster_info = []

    for label in range(CLUSTERS_NUM):
        subset = df_valid[df_valid['Cluster_Label'] == label]

        info = {
            'Label': label,
            'Center_Lat': subset['Lat_Decimal'].mean(),
            'Center_Lon': subset['Lon_Decimal'].mean(),
            'Start_Time': subset.index.min(),
            'End_Time': subset.index.max(),
            'Count': len(subset)
        }
        cluster_info.append(info)

    # 关键：按“开始时间”排序，理清浮标移动的时间线
    cluster_info.sort(key=lambda x: x['Start_Time'])

    print("\n" + "=" * 60)
    print(f"【浮标分段分析报告】(已过滤异常点 -> 聚类)")
    print("=" * 60)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝、橙、绿 (用于绘图)

    for i, info in enumerate(cluster_info):
        print(f"\n[阶段 {i + 1}] (Cluster {info['Label']})")
        print(f"  - 时间范围: {info['Start_Time']} 至 {info['End_Time']}")
        print(f"  - 样本数量: {info['Count']}")
        print(f"  - 中心坐标: Lat {info['Center_Lat']:.6f}, Lon {info['Center_Lon']:.6f}")
        print(f"  >>> 建议: GRAPES 提取坐标 -> ({info['Center_Lat']:.4f}, {info['Center_Lon']:.4f})")

        # 将颜色分配给 info 用于后续绘图
        info['Color'] = colors[i % len(colors)]

    # 获取最后一个阶段的中心，作为未来预测的基准
    last_phase = cluster_info[-1]
    print("-" * 60)
    print(f"\n>>> [未来预测建议]")
    print(f"请使用【阶段 {CLUSTERS_NUM}】的中心坐标: ({last_phase['Center_Lat']:.4f}, {last_phase['Center_Lon']:.4f})")

    # ========================================================
    # 【步骤 D】绘图验证
    # ========================================================
    plt.figure(figsize=(10, 8), dpi=120)

    # 1. 绘制所有有效点 (背景)
    # 为了看清分类，我们按聚类结果上色
    for info in cluster_info:
        label = info['Label']
        subset = df_valid[df_valid['Cluster_Label'] == label]

        plt.scatter(subset['Lon_Decimal'], subset['Lat_Decimal'],
                    s=20, alpha=0.4, c=info['Color'],
                    label=f"Phase {cluster_info.index(info) + 1}")

        # 绘制该阶段的中心
        plt.scatter(info['Center_Lon'], info['Center_Lat'],
                    s=300, marker='*', c=info['Color'], edgecolors='black',
                    zorder=10, label=f"Center {cluster_info.index(info) + 1}")

    plt.title(f'Buoy Movement Phases\n(Filter < {LIMIT_DEGREE}° -> K-Means K={CLUSTERS_NUM})')
    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')

    # 动态调整视野
    margin = 0.002
    plt.xlim(df_valid['Lon_Decimal'].min() - margin, df_valid['Lon_Decimal'].max() + margin)
    plt.ylim(df_valid['Lat_Decimal'].min() - margin, df_valid['Lat_Decimal'].max() + margin)
    plt.ticklabel_format(useOffset=False, style='plain')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()