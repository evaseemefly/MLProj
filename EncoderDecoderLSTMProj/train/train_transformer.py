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
    """加载所有站点数据，合并、排序、填充缺失值"""
    print("开始加载和整合数据...")
    all_dfs = []

    # TODO: 根据新的CONFIG结构，分别构建浮标和海洋站的路径并加载数据
    # 为了避免代码重复，我们先创建一个文件列表
    files_to_load = []

    # 1. 构建浮标文件列表
    buoy_base_path: Path = Path(config["data_path"]) / config["fub_relative_path"]
    for site_name in config["buoy_sites"]:
        files_to_load.append({
            "site_name": site_name,
            "path": str(Path(buoy_base_path) / f"2024_{site_name}_mergedata.h5"),
            "type": "浮标(buoy)"
        })

    # 2. 构建海洋站文件列表
    station_base_path: Path = Path(config["data_path"]) / config["station_relative_path"]
    for site_name in config["station_sites"]:
        files_to_load.append({
            "site_name": site_name,
            "path": str(Path(station_base_path) / f"2024_{site_name}_mergedata.h5"),
            "type": "海洋站(station)"
        })

    # 3. 统一循环加载所有文件
    for file_info in tqdm(files_to_load, desc="读取H5文件"):
        site_name = file_info["site_name"]
        file_path = file_info["path"]
        try:
            # TODO:[-] 25-10-28 修改为 使用 HDFStore 以只读模式('r')打开文件
            with pd.HDFStore(str(file_path), mode='r') as store:
                # 假设时间是Unix时间戳或可以转换为datetime的格式
                # 假设key是'data'，或者需要遍历找到
                # 这里我们假设文件内直接是数据集
                # 遍历文件中的每一个组（例如 '2024010100', '2024010112' ...）
                # 创建一个临时列表，用于存放当前文件内所有组的数据
                group_dfs = []
                for key in store.keys():
                    try:
                        # 访问当前组
                        group_df: pd.DataFrame = store[key]

                        # 从组内读取数据
                        # 假设时间仍然是Unix时间戳
                        # time = pd.to_datetime(group['time'][:], unit='s')
                        #
                        # # 为当前组的数据创建一个DataFrame
                        # df_group = pd.DataFrame({
                        #     'time': time,
                        #     f'{site_name}_real_u': group['realdata_u'][:],
                        #     f'{site_name}_real_v': group['realdata_v'][:],
                        #     f'{site_name}_forecast_u': group['forecast_u'][:],
                        #     f'{site_name}_forecast_v': group['forecast_v'][:],
                        # })
                        # --- 核心修改点 ---
                        # 校验索引是否为DatetimeIndex，如果不是则跳过
                        if not isinstance(group_df.index, pd.DatetimeIndex):
                            print(f"警告: 文件 {file_path} 组 {key} 的索引不是时间类型，已跳过。")
                            continue

                        group_dfs.append(group_df)

                    except KeyError as ke:
                        # 如果某个组内缺少某个数据集，打印错误并跳过这个组
                        print(f"在文件 {file_path} 的组 {key} 中读取失败，缺少键: {ke}")
                        continue

                # 如果成功读取了任何组的数据
                if group_dfs:
                    # 后续处理逻辑与之前版本相同，但现在更加可靠
                    # 1. 垂直合并文件内的所有组
                    single_site_df = pd.concat(group_dfs)
                    # 2. 按时间索引排序
                    single_site_df.sort_index(inplace=True)
                    # 3. 处理可能因合并产生的重复时间索引，保留最后一个值
                    single_site_df = single_site_df[~single_site_df.index.duplicated(keep='last')]
                    # 4. 重命名列以包含站点名
                    # 假设原始列名为 'realdata_u', 'realdata_v', 等
                    # 如果您的列名已经是唯一的，可以跳过此步，但为了通用性，保留此逻辑
                    rename_dict = {}
                    # 动态检查列是否存在，避免KeyError
                    for col in ['realdata_u', 'realdata_v', 'forecast_u', 'forecast_v']:
                        if col in single_site_df.columns:
                            rename_dict[col] = f'{site_name}_{col}'
                    single_site_df.rename(columns=rename_dict, inplace=True)

                    # 5. 将处理好的DataFrame添加到总列表中
                    all_dfs.append(single_site_df)
                else:
                    print(f"文件 {file_path} 中没有找到任何有效的数据组。")
        except Exception as e:
            print(f"读取 {file_info['type']} 文件 {file_path} 失败: {e}")
            # 根据需要决定是否因为一个文件失败而终止整个流程
            # return None

    if not all_dfs:
        print("没有成功加载任何数据，请检查文件路径和H5文件内容。")
        return None

    # 合并所有数据
    print("合并所有站点数据...")
    merged_df = pd.concat(all_dfs, axis=1)

    # 按时间排序并填充缺失值
    merged_df.sort_index(inplace=True)
    merged_df.fillna(method='ffill', inplace=True)
    merged_df.fillna(method='bfill', inplace=True)  # 填充开头可能存在的NaN

    if merged_df.isnull().sum().sum() > 0:
        print("警告：数据中仍存在缺失值，请检查数据源。")
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
