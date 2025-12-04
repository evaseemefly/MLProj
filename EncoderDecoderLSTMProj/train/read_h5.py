import pandas as pd

READ_PATH: str = r'H:\DATA\ML\MERGEDATA_H5\STATIONS\2024_BYQ_mergedata.h5'

def read_h5_file(read_path:str):
    # 使用上下文管理器打开，无需手动 close
    with pd.HDFStore(read_path, mode='r') as store:
        # 1. 查看文件里有哪些组/键 (Keys)
        print(store.keys())

        # 2. 读取某个具体的 DataFrame
        # keys: ['/20240101000000', '/20240101120000', '/20240102000000',]
        df = store['/20240101000000']

        """
                                       forecast_v  forecast_u  realdata_u  realdata_v
            2024-01-01 00:00:00+00:00    0.616738   -2.315477   -0.801398   -4.122834
            2024-01-01 03:00:00+00:00    3.488215    3.499143   -0.790064   -3.716961
            2024-01-01 06:00:00+00:00    6.546517    4.106787   -1.059839   -1.696096
            2024-01-01 09:00:00+00:00    6.357276    5.392305   -1.096255    0.488084
            2024-01-01 12:00:00+00:00    5.968379    5.781268    1.469463    2.022542
        """
        print(df.head())
        pass

if __name__ == '__main__':
    read_h5_file(READ_PATH)