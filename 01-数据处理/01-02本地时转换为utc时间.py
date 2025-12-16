import pandas as pd
import os

# ================= 配置区域 =================
# 需要处理的浮标列表
FUBS = ['MF01001', 'MF01002']
YEAR = 2024
# 数据所在的文件夹 (如果就在当前目录，保持 '.')
DATA_DIR = '.'


# ===========================================

def convert_csv_to_utc(buoy_id):
    # 原始文件名
    input_filename = os.path.join(DATA_DIR, f"{buoy_id}_{YEAR}_data.csv")
    # 新文件名
    output_filename = os.path.join(DATA_DIR, f"{buoy_id}_{YEAR}_data_UTC.csv")

    if not os.path.exists(input_filename):
        print(f"找不到文件: {input_filename}，跳过。")
        return

    print(f"正在转换: {input_filename} -> UTC ...")

    # 1. 读取 CSV
    # dtype={'TimeIndex': str} 至关重要，防止 '202401010000' 被识别为数字而丢失前导零或格式
    df = pd.read_csv(input_filename, index_col='TimeIndex', dtype={'TimeIndex': str})

    # 2. 解析时间索引 (北京时)
    # 格式为 YYYYMMDDHHMM
    bjt_time = pd.to_datetime(df.index, format='%Y%m%d%H%M')

    # 3. 转换为 UTC (减去 8 小时)
    utc_time = bjt_time - pd.Timedelta(hours=8)

    # 4. 更新索引并格式化回字符串
    # 依然保持 YYYYMMDDHHMM 格式，只是数值变了
    df.index = utc_time.strftime('%Y%m%d%H%M')

    # 5. 保存为新文件
    df.to_csv(output_filename, encoding='utf-8-sig')
    print(f"  -> 已保存: {output_filename}")

    # 可选：打印前几行对比一下
    # print(f"  [验证] BJT: {bjt_time[0]} -> UTC: {utc_time[0]}")


def main():
    for buoy in FUBS:
        convert_csv_to_utc(buoy)
    print("\n所有文件转换完成。")


if __name__ == "__main__":
    main()