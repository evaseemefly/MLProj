from pathlib import Path
import pandas as pd
import numpy as np
from private.dicts import dicts_name_code


def main():
    read_fullyear_fubs_dir: str = r'E:\01DATA\ML\FUB_RESAMPLED'
    read_fullyear_fubs_path: Path = Path(read_fullyear_fubs_dir)
    save_fullyear_fubs_path: Path = Path(read_fullyear_fubs_dir)
    # 获取目录下的所有文件并获取每个文件对应的code
    for temp_file in read_fullyear_fubs_path.rglob('*.csv'):
        temp_name: str = temp_file.name.split('.')[0].split('_')[2]
        temp_df = pd.read_csv(temp_file)
        temp_ws_list = temp_df['WS']
        temp_wd_list = temp_df['WD']
        # 存储为全年的 u | v 分量
        direction_rad = np.radians(temp_wd_list)
        # 2. 应用公式计算 u 和 v 分量
        # NumPy的sin和cos函数同样支持向量化操作
        u = -temp_ws_list * np.sin(direction_rad)
        v = -temp_ws_list * np.cos(direction_rad)

        # 3. 将计算出的 u, v 分量作为新列添加回 DataFrame
        # 这样可以保持数据结构的一致性
        target_u_stamp: str = f'u'
        target_v_stamp: str = f'v'
        temp_df[target_u_stamp] = u
        temp_df[target_v_stamp] = v
        saved_fill_path: Path = save_fullyear_fubs_path / f'{temp_name}_uv.csv'
        temp_df.to_csv(str(saved_fill_path), index=False)
        print(f'写入文件:{str(saved_fill_path)}成功~')


if __name__ == '__main__':
    main()
