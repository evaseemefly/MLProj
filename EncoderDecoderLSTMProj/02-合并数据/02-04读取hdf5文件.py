import pandas as pd
from pathlib import Path


def main(file_path: Path):
    try:
        # 使用 HDFStore 以只读模式('r')打开文件
        with pd.HDFStore(str(file_path), mode='r') as store:
            print("文件中的所有 keys 如下:")
            print(store.keys())
            print(f'读取的store中的keys的长度为:{len(store.keys())}')

            # 如果只有一个 key，可以直接读取
            if len(store.keys()) > 0:
                first_key = store.keys()[0]
                print(f"\n尝试读取第一个 key: '{first_key}'")
                df = store[first_key]  # 也可以用 store.get(first_key)
                print(df.head())

    except Exception as e:
        print(f"错误：文件 '{str(file_path)}' 未找到。")
        print(e)
        pass


if __name__ == '__main__':
    file_station: Path = Path(r'E:\01DATA\ML\MERGEDATA_H5\STATIONS\2024_BYQ_mergedata.h5')
    file_fub: Path = Path(r'E:\01DATA\ML\MERGEDATA_H5\FUB\2024_MF01002_mergedata.h5')
    main(file_fub)
    pass
