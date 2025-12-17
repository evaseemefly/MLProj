import os
import pandas as pd

# ================= 配置区域 =================

# 年份
YEAR = 2024


# ===========================================

def check_missing_files(out_put_file: str, wind_stamp: str = 'GRAPES'):
    # 1. 生成 2024 全年的标准时间序列 (频率: 12H, 即 00 和 12)
    # 2024是闰年，有366天
    # 开始时间: 2024-01-01 00:00 UTC
    # 结束时间: 2024-12-31 12:00 UTC
    expected_times = pd.date_range(
        start=f'{YEAR}-01-01 00:00',
        end=f'{YEAR}-12-31 12:00',
        freq='12H'
    )

    missing_records = []
    total_count = len(expected_times)

    print(f"开始检查 {DATA_DIR} 下的预报文件...")
    print(f"预期文件总数: {total_count} 个")

    # 2. 遍历检查
    for dt in expected_times:
        # 构建预期文件名: GRAPES_2024010100_240h_UV.nc
        # global_gfs_det_atm_2024123112.nc
        # zhongyuan_ecmwf_det_atm_2024123112.nc
        # dt.strftime('%Y%m%d%H') 会生成如 2024010100
        # file_name = f"{wind_stamp}_{dt.strftime('%Y%m%d%H')}_240h_UV.nc"
        file_name = f"zhongyuan_ecmwf_det_atm_{dt.strftime('%Y%m%d%H')}.nc"
        file_path = os.path.join(DATA_DIR, file_name)

        # 3. 判断文件是否存在
        if not os.path.exists(file_path):
            missing_records.append({
                'Missing_Time_UTC': dt,
                'Expected_Filename': file_name
            })
            # 可选: 打印每个缺失的文件
            # print(f"缺失: {file_name}")

    # 4. 统计与输出
    missing_count = len(missing_records)
    print("=" * 40)
    print(f"检查完成！")
    print(f"应有文件: {total_count}")
    print(f"实有文件: {total_count - missing_count}")
    print(f"缺失文件: {missing_count}")
    print("=" * 40)

    # 5. 保存到 CSV
    if missing_count > 0:
        df = pd.DataFrame(missing_records)
        df.to_csv(out_put_file, index=False, encoding='utf-8-sig')
        print(f"缺失列表已保存至: {os.path.abspath(out_put_file)}")

        # 预览前5个缺失
        print("\n缺失文件预览 (前5个):")
        print(df.head())
    else:
        print("完美！没有发现缺失文件。")


if __name__ == "__main__":

    wind_stamp = 'ECMWF'
    # 数据存储的根目录
    DATA_DIR = f'/Volumes/DATA/WIND/{wind_stamp}/2024'
    # 输出缺失列表的CSV文件名
    out_put_file = f'missing_{wind_stamp}_2024.csv'
    # 简单的路径检查
    if not os.path.exists(DATA_DIR):
        print(f"错误: 目录不存在 - {DATA_DIR}")
    else:
        check_missing_files(out_put_file, wind_stamp)
