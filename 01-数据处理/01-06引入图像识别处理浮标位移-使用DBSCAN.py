import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ================= 配置区域 =================
FILE_PATH = '/Volumes/WD_BLACK/DATA/CONVERTED/MF01001_2024_data_UTC.csv'
LIMIT_DEGREE = 0.03  # 空间过滤阈值

# DBSCAN 核心参数 (需要根据您的数据微调)
# eps: 两个点被视为“邻居”的最大距离 (单位: 度)
# 您的右边两团间隙看起来很小，约为 0.002-0.003度
# 所以 eps 设为 0.0015 左右应该能把它们分开
DBSCAN_EPS = 0.0008

# min_samples: 一个团至少包含多少个点
DBSCAN_MIN_SAMPLES = 50


# ===========================================

def dms_to_decimal(dms_str):
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


def main():
    print("1. 正在读取数据...")
    df = pd.read_csv(FILE_PATH, index_col='TimeIndex')
    df.index = pd.to_datetime(df.index, format='%Y%m%d%H%M')

    df['Lat_Decimal'] = df['Latitude'].apply(dms_to_decimal)
    df['Lon_Decimal'] = df['Longitude'].apply(dms_to_decimal)
    df_clean = df.dropna(subset=['Lat_Decimal', 'Lon_Decimal'])

    if len(df_clean) == 0:
        print("没有有效数据")
        return

    # 2. 空间过滤 (Filter)
    center_lat = df_clean['Lat_Decimal'].median()
    center_lon = df_clean['Lon_Decimal'].median()

    condition = (
            (np.abs(df_clean['Lat_Decimal'] - center_lat) < LIMIT_DEGREE) &
            (np.abs(df_clean['Lon_Decimal'] - center_lon) < LIMIT_DEGREE)
    )
    df_valid = df_clean[condition].copy()

    print(f"剩余有效点数: {len(df_valid)}")

    # ========================================================
    # 【核心改进】使用 DBSCAN 密度聚类
    # ========================================================
    print(f"正在执行 DBSCAN 聚类 (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")

    X = df_valid[['Lat_Decimal', 'Lon_Decimal']].values

    # 注意：DBSCAN 直接使用经纬度距离 (欧氏距离)
    # 如果数据量级差异大，通常需要 StandardScaler，但经纬度在这里量级一致，直接用即可物理意义更明确
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='euclidean')
    df_valid['Cluster_Label'] = db.fit_predict(X)

    # DBSCAN 会把噪声点标记为 -1，我们只保留非噪声点 (Label >= 0)
    # 如果您的 LIMIT_DEGREE 已经过滤得很好，这里应该没什么噪声
    n_clusters = len(set(df_valid['Cluster_Label'])) - (1 if -1 in df_valid['Cluster_Label'] else 0)
    n_noise = list(df_valid['Cluster_Label']).count(-1)

    print(f"\n>>> 自动识别出 {n_clusters} 个聚类簇 (Cluters)")
    print(f">>> 识别出 {n_noise} 个噪点 (Noise)")

    # ========================================================
    # 【分析结果】
    # ========================================================
    cluster_info = []

    # 获取唯一的标签 (排除 -1)
    unique_labels = set(df_valid['Cluster_Label'])
    if -1 in unique_labels: unique_labels.remove(-1)

    for label in unique_labels:
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

    # 按时间排序
    cluster_info.sort(key=lambda x: x['Start_Time'])

    print("\n" + "=" * 60)
    print(f"【DBSCAN 分段分析报告】")
    print("=" * 60)

    cmap = plt.get_cmap('tab10')

    for i, info in enumerate(cluster_info):
        print(f"\n[阶段 {i + 1}] (Label {info['Label']})")
        print(f"  - 时间范围: {info['Start_Time']} 至 {info['End_Time']}")
        print(f"  - 样本数量: {info['Count']}")
        print(f"  - 中心坐标: {info['Center_Lon']:.4f}, {info['Center_Lat']:.4f}")
        info['Color'] = cmap(i)

    # ========================================================
    # 【绘图】
    # ========================================================
    plt.figure(figsize=(10, 8), dpi=120)

    # 绘制正常簇
    for i, info in enumerate(cluster_info):
        label_id = info['Label']
        subset = df_valid[df_valid['Cluster_Label'] == label_id]

        plt.scatter(subset['Lon_Decimal'], subset['Lat_Decimal'],
                    s=20, alpha=0.4, c=[info['Color']],
                    label=f"Phase {i + 1}")

        plt.scatter(info['Center_Lon'], info['Center_Lat'],
                    s=300, marker='*', c=[info['Color']], edgecolors='black',
                    zorder=10)

    # 绘制 DBSCAN 判定为噪声的点 (如果有) - 用灰色小点表示
    noise_data = df_valid[df_valid['Cluster_Label'] == -1]
    if len(noise_data) > 0:
        plt.scatter(noise_data['Lon_Decimal'], noise_data['Lat_Decimal'],
                    s=10, c='gray', alpha=0.2, label='Noise (Ignored)')

    plt.title(f'DBSCAN Auto-Clustering (eps={DBSCAN_EPS})')
    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')

    margin = 0.002
    plt.xlim(df_valid['Lon_Decimal'].min() - margin, df_valid['Lon_Decimal'].max() + margin)
    plt.ylim(df_valid['Lat_Decimal'].min() - margin, df_valid['Lat_Decimal'].max() + margin)
    plt.ticklabel_format(useOffset=False, style='plain')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()