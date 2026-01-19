import pandas as pd
import xarray as xr
import numpy as np
import os

# ==========================================
# 1. 定义浮标阶段配置 (按优先级排序，越具体越靠后)
# ==========================================
# 注意：请仔细核对经纬度顺序，代码中统一为 Lat(纬度), Lon(经度)
# MF01002 和 MF01001 您提供的数据大概率是 (Lon, Lat) 格式 (120, 39)，此处已自动修正为 Lat, Lon

buoy_configs = {
    "MF01004": [
        {
            "start": "2024-06-27 11:00:00", "end": "2024-12-31 15:00:00",
            "lat": 37.9552, "lon": 119.8966
        }
    ],
    "MF01002": [
        # Stage 1 (通用大范围，先定义)
        {"start": "2023-12-31 16:00:00", "end": "2024-12-31 14:00:00", "lat": 39.0333, "lon": 120.0833},
        # Stage 2 (覆盖重叠区域)
        {"start": "2024-05-10 06:00:00", "end": "2024-12-31 15:00:00", "lat": 39.0350, "lon": 120.0833},
        # Stage 3 (更具体的覆盖)
        {"start": "2024-05-31 03:00:00", "end": "2024-12-31 09:00:00", "lat": 39.0333, "lon": 120.0850},
        # Stage 4 (极短时间)
        {"start": "2024-06-02 05:00:00", "end": "2024-06-02 05:00:00", "lat": 39.0350, "lon": 120.0850},
    ],
    "MF01001": [
        {"start": "2023-12-31 16:00:00", "end": "2024-09-05 00:00:00", "lat": 39.5005, "lon": 120.5951},
        {"start": "2024-09-05 01:00:00", "end": "2024-10-18 23:00:00", "lat": 39.5051, "lon": 120.6099},
        {"start": "2024-10-19 00:00:00", "end": "2024-12-31 15:00:00", "lat": 39.5031, "lon": 120.6094},
    ]
}


def apply_buoy_coordinates(df_obs, configs):
    """
    预处理：根据时间段给观测数据打上对应的 Lat/Lon
    """
    # 初始化目标列
    df_obs['Target_Lat'] = np.nan
    df_obs['Target_Lon'] = np.nan

    # 确保时间列是 datetime 类型
    df_obs['Obs_Time'] = pd.to_datetime(df_obs['Obs_Time'])

    print("正在根据阶段配置映射坐标...")

    for buoy_id, stages in configs.items():
        for stage in stages:
            start_t = pd.to_datetime(stage['start'])
            end_t = pd.to_datetime(stage['end'])

            # 筛选条件：ID匹配 且 时间在范围内
            mask = (
                    (df_obs['Buoy_ID'] == buoy_id) &
                    (df_obs['Obs_Time'] >= start_t) &
                    (df_obs['Obs_Time'] <= end_t)
            )

            # 赋值 (后定义的 Stage 会覆盖前面的)
            df_obs.loc[mask, 'Target_Lat'] = stage['lat']
            df_obs.loc[mask, 'Target_Lon'] = stage['lon']

    # 丢弃没有匹配到坐标的数据（可选）
    missing_coords = df_obs['Target_Lat'].isna().sum()
    if missing_coords > 0:
        print(f"警告: 有 {missing_coords} 条观测数据不在定义的阶段范围内，将被忽略。")
        df_obs = df_obs.dropna(subset=['Target_Lat'])

    return df_obs


def main():
    # ==========================================
    # 2. 主提取逻辑
    # ==========================================

    # 假设您已经加载了观测数据
    df_obs = pd.read_csv("your_buoy_data.csv")
    # 这里为了演示，手动创建一点假数据结构
    # data = {'Buoy_ID': ['MF01004', 'MF01002', 'MF01002'],
    #         'Obs_Time': ['2024-07-01 12:00:00', '2024-01-01 00:00:00', '2024-06-01 12:00:00']}
    # df_obs = pd.DataFrame(data)

    # 应用坐标映射
    df_obs = apply_buoy_coordinates(df_obs, buoy_configs)

    # 结果容器
    extraction_results = []

    # 获取文件列表
    nc_files = sorted([f for f in os.listdir('.') if f.startswith('GRAPES_') and f.endswith('.nc')])

    for nc_file in nc_files:
        # 1. 解析起报时间
        try:
            # 文件名示例: GRAPES_2024010112_240h_UV.nc
            str_time = nc_file.split('_')[1]
            run_time = pd.to_datetime(str_time, format='%Y%m%d%H')
        except Exception as e:
            print(f"文件名解析失败 {nc_file}: {e}")
            continue

        print(f"Processing Run Time: {run_time} ...")

        # 2. 打开 NetCDF
        with xr.open_dataset(nc_file) as ds:
            # 获取文件内的有效时间
            file_valid_times = pd.to_datetime(ds.time.values)

            # 3. 筛选当前文件能处理的观测数据
            # 只需要处理那些 Obs_Time 存在于当前文件 time 轴里的数据
            mask = df_obs['Obs_Time'].isin(file_valid_times)
            current_batch = df_obs[mask].copy()

            if current_batch.empty:
                continue

            # 4. 区域裁切 (Optimization: Crop)
            # 你的浮标大概在 Lat 37-40, Lon 119-121
            # 为了加快读取，先切一个小方块出来，避免读取全球/全国数据
            # 注意：这里加一点 buffer (36-42, 118-122) 确保浮标不跑出去
            mini_ds = ds.sel(
                latitude=slice(36, 42),  # 注意检查你的nc文件lat是升序还是降序，可能需要slice(42, 36)
                longitude=slice(118, 122)
            )

            # 5. 遍历该文件包含的每个有效时间步
            # 虽然可以直接用 xarray 的高级索引，但循环处理逻辑更清晰，且对于小数据量速度很快
            for valid_t in current_batch['Obs_Time'].unique():
                # 取出当前时刻需要提取的浮标记录
                targets = current_batch[current_batch['Obs_Time'] == valid_t]

                # 准备坐标数组 (xarray interp 需要 xarray 格式的坐标索引)
                target_lats = xr.DataArray(targets['Target_Lat'].values, dims="points")
                target_lons = xr.DataArray(targets['Target_Lon'].values, dims="points")

                # 计算预报时效
                lead_time = (valid_t - run_time).total_seconds() / 3600.0

                # --- 核心：空间插值提取 ---
                # 在切片后的小数据集上进行插值
                subset = mini_ds.sel(time=valid_t)

                # 使用 linear (双线性) 插值
                interpolated = subset.interp(
                    latitude=target_lats,
                    longitude=target_lons,
                    method='linear'
                )

                # 提取结果
                u_values = interpolated['UGRD_10maboveground'].values
                v_values = interpolated['VGRD_10maboveground'].values

                # 存入结果
                # 将 numpy 数组赋值回 DataFrame 切片 (保持顺序一致)
                targets = targets.copy()  # 避免 SettingWithCopyWarning
                targets['Model_U'] = u_values
                targets['Model_V'] = v_values
                targets['Run_Time'] = run_time
                targets['Lead_Time'] = lead_time

                extraction_results.append(targets)

    # ==========================================
    # 3. 合并与保存
    # ==========================================
    if extraction_results:
        final_df = pd.concat(extraction_results, ignore_index=True)

        # 清洗：去掉没有提取到数值的行（比如坐标超出范围）
        final_df = final_df.dropna(subset=['Model_U', 'Model_V'])

        # 导出
        final_df.to_csv("train_dataset_variable_coords.csv", index=False)
        print(f"提取完成，共生成 {len(final_df)} 条样本。")
    else:
        print("未提取到任何数据，请检查时间范围或文件名。")


if __name__ == '__main__':
    main()
