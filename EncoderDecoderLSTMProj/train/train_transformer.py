import math
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import re  # 引入正则表达式模块

# --- 1. 配置参数 ---
# TODO: 使用新CONFIG
CONFIG = {
    "data_path": r"E:/01DATA/ML/MERGEDATA_H5",  # 读取根目录
    # "data_path": r"/Volumes/WD_BLACK/ML/MERGEDATA_H5",  # 读取根目录
    "fub_relative_path": "FUB",  # 浮标相对路径
    "station_relative_path": "STATIONS",  # 海洋站相对路径
    # "buoy_sites": ['MF01002', 'MF01004', 'MF02001', 'MF02004'],  # 浮标站文件名（不含.h5）
    "buoy_sites": ['MF01002', 'MF01004'],  # 浮标站文件名（不含.h5）
    # "station_sites": ['BYQ', 'CFD', 'CST', 'DGG', 'LHT', 'PLI', 'QHD', 'TGU', 'WFG'],  # 海洋站文件名
    "station_sites": ['BYQ', 'CST', 'DGG'],  # 海洋站文件名

    # 序列长度
    "encoder_seq_len": 24,  # 过去24小时
    "decoder_seq_len": 24,  # 未来24小时

    # 模型参数
    "d_model": 128,  # 模型隐藏维度
    "nhead": 8,  # Transformer多头注意力头数
    "num_encoder_layers": 4,
    "num_decoder_layers": 4,
    "dim_feedforward": 512,
    "dropout": 0.1,

    # 训练参数
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 0.0001,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
# 计算总站点数和特征数
CONFIG["all_sites"] = CONFIG["buoy_sites"] + CONFIG["station_sites"]
NUM_SITES = len(CONFIG["all_sites"])
# 编码器输入特征: realdata_u, realdata_v, forecast_u, forecast_v
CONFIG["encoder_feature_num"] = NUM_SITES * 4
# 解码器输入特征: forecast_u, forecast_v
CONFIG["decoder_feature_num"] = NUM_SITES * 2
# 输出/目标特征: realdata_u, realdata_v
CONFIG["output_feature_num"] = NUM_SITES * 2


def parse_issuance_time_from_key(key):
    """
    从HDFStore的key (例如 '/fc_2024010100') 中解析出发布时间。
    返回一个pandas Timestamp对象。
    """
    # 使用正则表达式查找连续的数字，假设格式是 YYYYMMDDHH
    match = re.search(r'(\d{10})', key)
    if match:
        try:
            # 解析为 datetime 对象，然后转换为带时区的 Timestamp
            dt = pd.to_datetime(match.group(1), format='%Y%m%d%H')
            return pd.Timestamp(dt, tz='UTC')  # 假设是UTC时间
        except ValueError:
            return None  # 如果格式不匹配，返回None
    return None


def parse_column_names(columns):
    """
    辅助函数：预解析所有列名，以提高主循环的效率。
    返回一个包含列信息的DataFrame，并按发布时间排序。
    """
    print("正在预解析列名以提高效率...")
    parsed_cols = []
    # 正则表达式匹配 '站点_类型_变量_issued_时间戳' 格式
    pattern = re.compile(r'(.+?)_(forecast|realdata)_(u|v)_issued_(.+)')

    for col in columns:
        match = pattern.match(col)
        if match:
            site, type, var, issue_time_str = match.groups()
            issue_time = pd.to_datetime(issue_time_str)
            parsed_cols.append({
                'col_name': col,
                'site': site,
                'type': type,
                'var': var,
                'issue_time': issue_time
            })

    if not parsed_cols:
        raise ValueError("无法从列名中解析出任何有效信息。请检查列名格式。")

    df = pd.DataFrame(parsed_cols)
    # 按发布时间排序，这对于后续查找“最新”预报至关重要
    df.sort_values('issue_time', inplace=True)
    return df


# --- 2. Transformer 模型定义 ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    def __init__(self, config):
        super(TimeSeriesTransformer, self).__init__()
        self.config = config

        self.encoder_embedding = nn.Linear(config["encoder_feature_num"], config["d_model"])
        self.decoder_embedding = nn.Linear(config["decoder_feature_num"], config["d_model"])

        self.pos_encoder = PositionalEncoding(config["d_model"], config["dropout"])

        self.transformer = nn.Transformer(
            d_model=config["d_model"],
            nhead=config["nhead"],
            num_encoder_layers=config["num_encoder_layers"],
            num_decoder_layers=config["num_decoder_layers"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"],
            batch_first=True
        )

        self.fc_out = nn.Linear(config["d_model"], config["output_feature_num"])

    def forward(self, src, tgt):
        src = self.encoder_embedding(src) * math.sqrt(self.config["d_model"])
        src = self.pos_encoder(src.transpose(0, 1)).transpose(0, 1)

        tgt = self.decoder_embedding(tgt) * math.sqrt(self.config["d_model"])
        tgt = self.pos_encoder(tgt.transpose(0, 1)).transpose(0, 1)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(self.config["device"])

        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        output = self.fc_out(output)

        return output


# --- 3. 数据集定义 ---

class MultiSiteDataset(Dataset):
    def __init__(self, encoder_x, decoder_x, target_y):
        self.encoder_x = torch.FloatTensor(encoder_x)
        self.decoder_x = torch.FloatTensor(decoder_x)
        self.target_y = torch.FloatTensor(target_y)

    def __len__(self):
        return len(self.encoder_x)

    def __getitem__(self, idx):
        return self.encoder_x[idx], self.decoder_x[idx], self.target_y[idx]


# --- 4. 数据处理函数 ---
def load_and_prepare_data(config):
    """
        加载所有站点数据，将不同发布时间的预报横向拼接，然后合并所有站点。
        步骤：
            加载和构建宽表
        TODO:[*] 25-11-03 修改了之前按照index进行纵向拼接并去重的逻辑错误。
                     新策略：对单个文件内的不同发布时间(group)进行横向拼接。

    :param config:
    :return:
    """
    # TODO:[*] 修改了打印信息，以反映新的处理策略
    print("开始加载和整合数据 (V3 - 横向拼接策略)...")
    all_dfs = []

    # 文件列表构建逻辑保持不变
    files_to_load = []
    buoy_base_path: Path = Path(config["data_path"]) / config["fub_relative_path"]
    for site_name in config["buoy_sites"]:
        files_to_load.append({
            "site_name": site_name,
            "path": str(Path(buoy_base_path) / f"2024_{site_name}_mergedata.h5"),
            "type": "浮标(buoy)"
        })

    station_base_path: Path = Path(config["data_path"]) / config["station_relative_path"]
    for site_name in config["station_sites"]:
        files_to_load.append({
            "site_name": site_name,
            "path": str(Path(station_base_path) / f"2024_{site_name}_mergedata.h5"),
            "type": "海洋站(station)"
        })

    # TODO:[*] 修改了tqdm的描述信息
    for file_info in tqdm(files_to_load, desc="读取H5文件并横向拼接"):
        site_name = file_info["site_name"]
        file_path = file_info["path"]
        try:
            with pd.HDFStore(str(file_path), mode='r') as store:
                # TODO:[*] 逻辑重构：不再使用 group_dfs 进行纵向拼接
                # 创建一个新列表，用于存放当前文件内所有经过重命名的组DataFrame
                site_forecast_dfs = []

                if not store.keys():
                    print(f"警告: 文件 {file_path} 为空或不包含任何组。")
                    continue

                for key in store.keys():
                    try:
                        # TODO:[*] 1. 解析发布时间
                        issuance_time_str = parse_issuance_time_from_key(key)
                        if issuance_time_str is None:
                            print(f"警告: 无法从组名 '{key}' (文件: {file_path}) 解析发布时间，已跳过。")
                            continue

                        group_df: pd.DataFrame = store[key]

                        if not isinstance(group_df.index, pd.DatetimeIndex):
                            print(f"警告: 文件 {file_path} 组 {key} 的索引不是时间类型，已跳过。")
                            continue

                        # TODO:[*] 2. 创建唯一的列名，包含发布时间信息
                        # 例如: 'realdata_u' -> 'realdata_u_issued_2024010100'
                        rename_dict = {
                            col: f'{col}_issued_{issuance_time_str}'
                            for col in group_df.columns
                        }
                        group_df_renamed = group_df.rename(columns=rename_dict)

                        # TODO:[*] 3. 将重命名后的DataFrame添加到待拼接列表
                        site_forecast_dfs.append(group_df_renamed)

                    except KeyError as ke:
                        print(f"在文件 {file_path} 的组 {key} 中读取失败，缺少键: {ke}")
                        continue

                # TODO:[*] 4. 逻辑重构：处理单个站点的所有预报
                if site_forecast_dfs:
                    # 4.1. 横向拼接单个站点的所有预报。Pandas会根据索引（预报有效时间）自动对齐
                    # TODO:[*] 25-11-05
                    # shape:(2944, 2800)
                    single_site_df = pd.concat(site_forecast_dfs, axis=1)

                    # 4.2. 为所有列添加站点名前缀，这是最终的列名
                    # 例如: 'realdata_u_issued_2024010100' -> 'MF01002_realdata_u_issued_2024010100'
                    single_site_df = single_site_df.add_prefix(f'{site_name}_')

                    # 4.3. 将处理好的单个站点的完整DataFrame添加到总列表中
                    all_dfs.append(single_site_df)
                else:
                    print(f"文件 {file_path} 中没有找到任何有效的数据组。")
        except Exception as e:
            print(f"读取 {file_info['type']} 文件 {file_path} 失败: {e}")
            continue

    if not all_dfs:
        print("没有成功加载任何数据，请检查文件路径和H5文件内容。")
        return None

    # 合并所有数据
    print("合并所有站点数据...")
    # TODO:[*] 这里的 axis=1 现在合并的是每个站点的宽表，功能正确，无需修改，但意义已变。
    merged_df = pd.concat(all_dfs, axis=1)

    # 按时间排序并填充缺失值 (这部分逻辑保持不变，依然适用)
    # TODO:[*] 25-11-05 shape: (2944, 14000)
    merged_df.sort_index(inplace=True)

    # TODO:[-] 25-11-05 原始数据保留了每个预报的独立性和有效时间窗口，不应该进行fillna，会制造虚假数据，破坏特征的意义，此处不再进行缺省值的填充
    # print("填充缺失值 (forward fill)...")
    # merged_df.fillna(method='ffill', inplace=True)
    # print("填充缺失值 (backward fill)...")
    # merged_df.fillna(method='bfill', inplace=True)
    #
    # if merged_df.isnull().sum().sum() > 0:
    #     print("警告：数据填充后仍存在缺失值，将用0填充。")
    #     merged_df.fillna(0, inplace=True)

    print("数据整合完成！")
    return merged_df


def create_samples(merged_df: pd.DataFrame, config: dict):
    """
            从原始宽表中创建特征，并使用滑动窗口生成训练样本。
    @param merged_df: - merged_df (pd.DataFrame): 从 load_and_prepare_data 得到的原始宽表。
    @param config:     - config (dict): 包含 all_sites, encoder_seq_len, decoder_seq_len 等配置的字典。
    @return:
    - encoder_x (np.array): 编码器输入序列。
    - decoder_x (np.array): 解码器输入序列。
    - target_y (np.array): 目标序列。
    - scaler (StandardScaler): 用于数据标准化的 scaler 对象。
    - feature_df.columns (pd.Index): 返回特征列名，便于后续分析。
    """

    # =========================================================================
    # Part 1: 特征工程 - 将原始宽表转换为干净的特征表
    # =========================================================================
    print("Part 1: 开始特征工程，从原始数据中提取干净特征...")

    # 1.1 预解析列名以提高效率
    col_info_df = parse_column_names(merged_df.columns)

    # 1.2 创建一个空的特征DataFrame，索引与原始df一致
    feature_df = pd.DataFrame(index=merged_df.index)

    # 1.3 遍历每个站点，为其提取真实值和最新的预报值
    for site in tqdm(config["all_sites"], desc="为每个站点提取特征"):
        site_cols_info = col_info_df[col_info_df['site'] == site].copy()

        # --- 提取真实值 (realdata) ---
        """
            处理真实值与处理预报数据处理方式不同
            对于真实值：
                对于同一个时间点（例如 2024-01-01 12:00:00），它的真实风速值是唯一的、确定的。
                理论上，这个值在上述三个文件中都应该被记录，且完全相同。
        """
        for var in ['u', 'v']:
            real_cols = site_cols_info[(site_cols_info['type'] == 'realdata') & (site_cols_info['var'] == var)][
                'col_name']
            if not real_cols.empty:
                # 使用 bfill(axis=1) 高效地将每行的第一个非NaN值填充到整行，然后取第一列。
                # TODO:[*] 25-11-06 注意此处可能存在问题：若第一列的实况数据并不完整，若后面几列有实况值，应如何处理
                """
                    merged_df[real_cols]：选出所有相关的真实数据列。对于 2024-01-01 10:00:00 这一行，数据是 [5.1, NaN, NaN]。
                    .bfill(axis=1)：水平向后填充。[5.1, NaN, NaN] 保持不变，因为第一个就是有效值。
                    .iloc[:, 0]：取第一列。结果是 5.1。
                    
                    对于 2024-01-02 08:00:00 这一行，数据是 [NaN, 4.8, 4.8]。                    
                    .bfill(axis=1)：水平向后填充，[NaN, 4.8, 4.8] 变为 [4.8, 4.8, 4.8]。
                    .iloc[:, 0]：取第一列。结果是 4.8。
                """
                feature_df[f'{site}_real_{var}'] = merged_df[real_cols].bfill(axis=1).iloc[:, 0]

        # --- 提取最新的预报值 (forecast) ---
        # 这是一个高效的向量化实现，避免了逐行循环。
        for var in ['u', 'v']:
            forecast_cols_info = site_cols_info[(site_cols_info['type'] == 'forecast') & (site_cols_info['var'] == var)]

            if forecast_cols_info.empty:
                # 如果该站点没有预报数据，则创建一个全为NaN的列
                feature_df[f'{site}_forecast_{var}'] = np.nan
                continue

            # 创建一个空的Series来存放每个时间点的最新预报
            latest_forecast = pd.Series(np.nan, index=merged_df.index)

            # 遍历按发布时间排序的预报列
            for _, row in forecast_cols_info.iterrows():
                col_name = row['col_name']
                # 使用 update 方法。由于我们是按发布时间从旧到新遍历，
                # 后续的（更新的）预报会覆盖掉旧的预报值。
                # 最终，在每个时间点上留下的就是最新的有效预报。
                latest_forecast.update(merged_df[col_name])

            feature_df[f'{site}_forecast_{var}'] = latest_forecast

    print(f"\n特征工程完成。创建的特征表 feature_df 的形状: {feature_df.shape}")
    print(f"特征列示例: {feature_df.columns[:4].tolist()}...")

    # =========================================================================
    # Part 2: 滑动窗口 - 在干净的特征表上创建样本
    # =========================================================================
    print("\nPart 2: 开始创建滑动窗口样本...")

    # 2.1 处理缺失值
    # 在滑动窗口前，用0填充所有NaN。更复杂的策略（如插值）也可以在这里应用。
    feature_df.fillna(0, inplace=True)

    # TODO:[*] 25-11-10 生成 编码器 | 解码器 list 均需要通过 滑动窗口长度 遍历 截取 scaled_data。scaled_data的作用是什么？ (2944, 20)
    # 2.2 数据标准化
    # 它会改变 feature_df 中每个特征（每一列）的分布，使其均值为0，标准差为1。
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(feature_df)

    # 2.3 定义编码器、解码器和目标的特征列
    all_sites = config["all_sites"]
    """
        编码器  : 预报 | 实况
        => 
        解码器  : 预报
        =>
        目标值  : 实况
    """

    """
        ['MF01002_real_u', 'MF01002_real_v', 'MF01002_forecast_u', 'MF01002_forecast_v',
         'MF01004_real_u', 'MF01004_real_v', 'MF01004_forecast_u', 'MF01004_forecast_v',
          'BYQ_real_u', 'BYQ_real_v', 'BYQ_forecast_u', 'BYQ_forecast_v', 
          'CST_real_u', 'CST_real_v', 'CST_forecast_u', 'CST_forecast_v', 
          'DGG_real_u', 'DGG_real_v', 'DGG_forecast_u', 'DGG_forecast_v']
          ————————————————————————————————————————
          编码器特征值 包含 实况数据 与 预报数据
    """
    encoder_features = [f"{site}_{ftype}_{var}" for site in all_sites for ftype in ["real", "forecast"] for var in
                        ["u", "v"]]
    """
        ['MF01002_forecast_u', 'MF01002_forecast_v', 
        'MF01004_forecast_u', 'MF01004_forecast_v', 
        'BYQ_forecast_u', 'BYQ_forecast_v', 
        'CST_forecast_u', 'CST_forecast_v', 
        'DGG_forecast_u', 'DGG_forecast_v']
        ————————————————————————————————————————
        解码器器特征值 包含 预报数据
    """
    decoder_features = [f"{site}_forecast_{var}" for site in all_sites for var in ["u", "v"]]
    """
        ['MF01002_real_u', 'MF01002_real_v', 
        'MF01004_real_u', 'MF01004_real_v', 
        'BYQ_real_u', 'BYQ_real_v', 
        'CST_real_u', 'CST_real_v',
         'DGG_real_u', 'DGG_real_v']
         ————————————————————————————————————————
         目标特征值 包含 实况数据
    """
    target_features = [f"{site}_real_{var}" for site in all_sites for var in ["u", "v"]]

    # 获取这些特征在 scaled_data 中的列索引

    """
        ['MF01002_real_u', 'MF01002_real_v','MF01002_forecast_u', 'MF01002_forecast_v', 
        'MF01004_real_u', 'MF01004_real_v', 'MF01004_forecast_u', 'MF01004_forecast_v', 
        'BYQ_real_u', 'BYQ_real_v', 'BYQ_forecast_u', 'BYQ_forecast_v', 
        'CST_real_u', 'CST_real_v', 'CST_forecast_u', 'CST_forecast_v', 
        'DGG_real_u', 'DGG_real_v', 'DGG_forecast_u', 'DGG_forecast_v']
    """
    df_cols = feature_df.columns.tolist()

    # 以下为 编码器 | 解码器 | 目标值
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    encoder_indices = [df_cols.index(col) for col in encoder_features]
    # [2, 3, 6, 7, 10, 11, 14, 15, 18, 19]
    decoder_indices = [df_cols.index(col) for col in decoder_features]
    # [0, 1, 4, 5, 8, 9, 12, 13, 16, 17]
    target_indices = [df_cols.index(col) for col in target_features]

    # TODO:[*] 25-11-10 滑动窗口设置为 24 是否合适？可否延长，而且预报时间间隔也不是1小时，是3小时。
    # 2.4 创建滑动窗口样本
    encoder_seq_len = config["encoder_seq_len"]
    decoder_seq_len = config["decoder_seq_len"]

    encoder_x_list, decoder_x_list, target_y_list = [], [], []
    # 2944
    total_len = len(scaled_data)
    # 48
    window_len = encoder_seq_len + decoder_seq_len

    for i in tqdm(range(total_len - window_len + 1), desc="生成样本"):
        # 编码器输入：历史数据
        encoder_start = i
        encoder_end = i + encoder_seq_len
        encoder_x_list.append(scaled_data[encoder_start:encoder_end, encoder_indices])

        # 解码器输入：未来的预报数据
        decoder_start = encoder_end - 1  # 从编码器最后一步开始
        decoder_end = decoder_start + decoder_seq_len
        decoder_x_list.append(scaled_data[decoder_start:decoder_end, decoder_indices])

        # 目标：未来的真实数据
        target_start = encoder_end
        target_end = target_start + decoder_seq_len
        target_y_list.append(scaled_data[target_start:target_end, target_indices])

    # 2.5 将列表转换为Numpy数组
    encoder_x = np.array(encoder_x_list)
    decoder_x = np.array(decoder_x_list)
    target_y = np.array(target_y_list)

    print(f"\n样本创建完成。")
    print(f"编码器输入 (Encoder X) 形状: {encoder_x.shape}")
    print(f"解码器输入 (Decoder X) 形状: {decoder_x.shape}")
    print(f"目标 (Target Y) 形状: {target_y.shape}")

    return encoder_x, decoder_x, target_y, scaler, feature_df.columns


# --- 5. 主训练流程 ---
def main():
    print(f"使用设备: {CONFIG['device']}")

    # 1. 加载和准备数据
    merged_df = load_and_prepare_data(CONFIG)
    if merged_df is None:
        return

    # 2. 创建样本
    encoder_x, decoder_x, target_y, scaler = create_samples(merged_df, CONFIG)

    # 3. 划分数据集
    (enc_x_train, enc_x_val,
     dec_x_train, dec_x_val,
     y_train, y_val) = train_test_split(
        encoder_x, decoder_x, target_y, test_size=0.2, random_state=42
    )

    train_dataset = MultiSiteDataset(enc_x_train, dec_x_train, y_train)
    val_dataset = MultiSiteDataset(enc_x_val, dec_x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # 4. 初始化模型、损失函数、优化器
    model = TimeSeriesTransformer(CONFIG).to(CONFIG["device"])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    print("开始训练模型...")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}")

        for src, tgt, target in progress_bar:
            src, tgt, target = src.to(CONFIG["device"]), tgt.to(CONFIG["device"]), target.to(CONFIG["device"])

            optimizer.zero_grad()

            output = model(src, tgt)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt, target in val_loader:
                src, tgt, target = src.to(CONFIG["device"]), tgt.to(CONFIG["device"]), target.to(CONFIG["device"])
                output = model(src, tgt)
                loss = criterion(output, target)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

    # 6. 保存模型
    torch.save(model.state_dict(), "multi_site_transformer.pth")
    print("模型已保存到 multi_site_transformer.pth")


if __name__ == "__main__":
    main()
