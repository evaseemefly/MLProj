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
    "fub_relative_path": "FUB",  # 浮标相对路径
    "station_relative_path": "STATIONS",  # 海洋站相对路径
    "buoy_sites": ['MF01002', 'MF01004', 'MF02001', 'MF02004'],  # 浮标站文件名（不含.h5）
    "station_sites": ['BYQ', 'CFD', 'CST', 'DGG', 'LHT', 'PLI', 'QHD', 'TGU', 'WFG'],  # 海洋站文件名

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
    merged_df.sort_index(inplace=True)
    print("填充缺失值 (forward fill)...")
    merged_df.fillna(method='ffill', inplace=True)
    print("填充缺失值 (backward fill)...")
    merged_df.fillna(method='bfill', inplace=True)

    if merged_df.isnull().sum().sum() > 0:
        print("警告：数据填充后仍存在缺失值，将用0填充。")
        merged_df.fillna(0, inplace=True)

    print("数据整合完成！")
    return merged_df

def create_samples(df, config):
    """使用滑动窗口创建训练样本"""
    print("开始创建滑动窗口样本...")

    # 定义特征列
    encoder_cols = []
    decoder_cols = []
    target_cols = []
    for site in config["all_sites"]:
        encoder_cols.extend([f'{site}_real_u', f'{site}_real_v', f'{site}_forecast_u', f'{site}_forecast_v'])
        decoder_cols.extend([f'{site}_forecast_u', f'{site}_forecast_v'])
        target_cols.extend([f'{site}_real_u', f'{site}_real_v'])

    # 数据标准化
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, index=df.index, columns=df.columns)

    encoder_x, decoder_x, target_y = [], [], []

    total_len = len(scaled_df)
    enc_len = config["encoder_seq_len"]
    dec_len = config["decoder_seq_len"]

    for i in tqdm(range(total_len - enc_len - dec_len + 1), desc="生成样本"):
        encoder_start = i
        encoder_end = i + enc_len
        decoder_start = encoder_end
        decoder_end = encoder_end + dec_len

        encoder_x.append(scaled_df.iloc[encoder_start:encoder_end][encoder_cols].values)
        decoder_x.append(scaled_df.iloc[decoder_start:decoder_end][decoder_cols].values)
        target_y.append(scaled_df.iloc[decoder_start:decoder_end][target_cols].values)

    print(f"样本生成完毕，共 {len(encoder_x)} 个样本。")
    return np.array(encoder_x), np.array(decoder_x), np.array(target_y), scaler


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
