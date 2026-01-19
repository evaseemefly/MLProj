import pandas as pd


def read_merge_df(csv_path: str) -> pd.DataFrame:
    # 1. 读取您已经合并好的大 CSV

    df_all = pd.read_csv(csv_path)

    # 确保时间格式正确
    df_all['Obs_Time'] = pd.to_datetime(df_all['Obs_Time'])
    return df_all


def merge2h5(df_all: pd.DataFrame, h5_path: str):
    # 2. 定义 HDF5 文件名

    # 3. 遍历每个浮标并存储
    # 使用 HDFStore 上下文管理器，安全高效
    with pd.HDFStore(h5_path, mode='w', complib='blosc', complevel=9) as store:
        # 获取所有唯一的浮标 Code
        unique_buoys = df_all['Buoy_ID'].unique()

        for buoy_id in unique_buoys:
            print(f"正在存储浮标: {buoy_id} ...")

            # 筛选出该浮标的数据
            df_sub = df_all[df_all['Buoy_ID'] == buoy_id].copy()

            # 【优化建议】：存入前把时间设为索引并排序
            # 这样读出来直接就是时间序列，非常方便后续插值
            df_sub = df_sub.set_index('Obs_Time').sort_index()

            # 存入 H5，Key 就是浮标 ID
            # format='table' 支持后续在不读取文件的情况下进行查询，但在我们这种拆分 key 的场景下，
            # format='fixed' (默认) 读写速度更快。这里用 fixed 即可。
            store.put(key=buoy_id, value=df_sub, format='fixed')

    print("H5 文件创建完成！")


def main():
    csv_path = r"/Volumes/WD_BLACK/DATA/EXPORT/buoy_obs_all_cleaned.csv"
    h5_path = r"/Volumes/WD_BLACK/DATA/EXPORT/buoy_obs_all_cleaned.h5"
    df_all_csv = read_merge_df(csv_path)
    merge2h5(df_all_csv, h5_path)
    pass


if __name__ == '__main__':
    main()
