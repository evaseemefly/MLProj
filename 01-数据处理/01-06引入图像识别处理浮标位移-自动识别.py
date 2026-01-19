import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ================= 配置区域 =================
# FILE_PATH = '/Volumes/WD_BLACK/DATA/CONVERTED/MF01001_2024_data_UTC.csv'
FILE_PATH = '/Volumes/WD_BLACK/DATA/CONVERTED/MF01004_2024_data_UTC.csv'
LIMIT_DEGREE = 0.03  # 空间过滤阈值
MAX_K_TO_TEST = 6  # 自动测试的最大簇数量（比如测试2到6）


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

    # 2. 坐标转换
    df['Lat_Decimal'] = df['Latitude'].apply(dms_to_decimal)
    df['Lon_Decimal'] = df['Longitude'].apply(dms_to_decimal)
    df_clean = df.dropna(subset=['Lat_Decimal', 'Lon_Decimal'])

    if len(df_clean) == 0:
        print("没有有效数据")
        return

    # ========================================================
    # 【步骤 A】空间过滤 (剔除极端离群点)
    # ========================================================
    center_lat = df_clean['Lat_Decimal'].median()
    center_lon = df_clean['Lon_Decimal'].median()

    condition = (
            (np.abs(df_clean['Lat_Decimal'] - center_lat) < LIMIT_DEGREE) &
            (np.abs(df_clean['Lon_Decimal'] - center_lon) < LIMIT_DEGREE)
    )
    df_valid = df_clean[condition].copy()

    print(f"已剔除 {len(df_clean) - len(df_valid)} 个极端漂移点")
    print(f"剩余有效点数: {len(df_valid)}")

    # ========================================================
    # 【步骤 B】自动寻找最佳 K 值 (基于轮廓系数)
    # ========================================================
    print(f"\n正在自动评估最佳聚类数量 (测试范围 K=2 到 {MAX_K_TO_TEST})...")

    X = df_valid[['Lat_Decimal', 'Lon_Decimal']].values

    best_k = 2
    best_score = -1
    scores = []

    # 遍历测试不同的 K 值
    for k in range(2, MAX_K_TO_TEST + 1):
        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans_test.fit_predict(X)

        # 计算轮廓系数 (-1 到 1，越接近 1 越好)
        score = silhouette_score(X, labels)
        scores.append((k, score))
        print(f"  - K={k}: 轮廓系数 Score = {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"\n>>> 自动判定最佳聚类数量为: {best_k} 团 (Score: {best_score:.4f})")

    # ========================================================
    # 【步骤 C】使用最佳 K 值执行聚类
    # ========================================================
    optimal_k = best_k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_valid['Cluster_Label'] = kmeans.fit_predict(X)

    # ========================================================
    # 【步骤 D】分析结果与绘图
    # ========================================================
    cluster_info = []
    for label in range(optimal_k):
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
    print(f"【浮标分段分析报告】(自动识别为 {optimal_k} 个阶段)")
    print("=" * 60)

    # 动态生成足够多的颜色
    cmap = plt.get_cmap('tab10')

    for i, info in enumerate(cluster_info):
        print(f"\n[阶段 {i + 1}]")
        print(f"  - 时间范围: {info['Start_Time']} 至 {info['End_Time']}")
        print(f"  - 样本数量: {info['Count']}")
        print(f"  - 中心坐标: {info['Center_Lon']:.4f}, {info['Center_Lat']:.4f}")

        # 分配颜色
        info['Color'] = cmap(i)

    # 绘图
    plt.figure(figsize=(10, 8), dpi=120)
    for info in cluster_info:
        label_id = info['Label']
        subset = df_valid[df_valid['Cluster_Label'] == label_id]

        # 散点
        plt.scatter(subset['Lon_Decimal'], subset['Lat_Decimal'],
                    s=20, alpha=0.4, c=[info['Color']],
                    label=f"Phase {cluster_info.index(info) + 1}")

        # 中心
        plt.scatter(info['Center_Lon'], info['Center_Lat'],
                    s=300, marker='*', c=[info['Color']], edgecolors='black',
                    zorder=10)

    plt.title(f'Auto-Detected Phases (Best K={optimal_k}, Score={best_score:.2f})')
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