from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns

# 引入您原来的代码以便复用配置和模型类
# 假设您的训练脚本名为 train_transformer.py
from train_transformer import CONFIG, MODEL_PATH, load_and_prepare_data, create_samples, TimeSeriesTransformer, \
    MultiSiteDataset
from torch.utils.data import DataLoader

# 设置字体以支持中文显示 (根据您的系统调整)
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
plt.rcParams['axes.unicode_minus'] = False


def inverse_transform_data(scaler, data_scaled, feature_indices, total_features):
    """
    辅助函数：将部分特征的反归一化。
    StandardScaler 需要输入的形状与 fit 时一致 (N, total_features)。
    但我们的 data_scaled 通常只有部分列（比如只有实况u,v）。
    """
    # 创建一个全 0 的 dummy 矩阵
    N = data_scaled.shape[0]
    dummy = np.zeros((N, total_features))

    # 将已知的数据填入对应的列
    # data_scaled 可能是 (N, seq_len, features) 或者 (N, features)
    # 这里处理平铺后的数据 (N*seq_len, features)
    dummy[:, feature_indices] = data_scaled

    # 反归一化
    inversed_dummy = scaler.inverse_transform(dummy)

    # 取出我们需要的列
    return inversed_dummy[:, feature_indices]


def evaluate():
    # 1. 准备数据 (必须与训练时完全一致，以保证 Scaler 相同)
    print("正在加载数据用于评估...")
    # merged_df = load_and_prepare_data(CONFIG)
    # # 注意：这里会重新生成 scaler，必须保证逻辑与训练时一致
    # encoder_x, decoder_x, target_y, scaler, columns = create_samples(merged_df, CONFIG)
    #
    # # 划分出验证集 (取后20%)
    # train_size = int(len(encoder_x) * 0.8)
    #
    # # 我们只评估验证集的数据
    # test_enc = encoder_x[train_size:]
    # test_dec = decoder_x[train_size:]
    # test_y = target_y[train_size:]
    #
    # # 转为 Tensor
    # dataset = MultiSiteDataset(test_enc, test_dec, test_y)
    # loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. 加载模型
    # 注意评估时在跨平台设备进行评估会出现如下错误:
    """
        原因： 这个模型权重文件（.pth）是在一台有 NVIDIA 显卡（CUDA） 的机器上保存的，而你现在的 M1 Pro Mac 并不支持 CUDA（它使用的是 MPS 或 CPU）。
        PyTorch 默认会尝试把模型加载回它当初保存时的设备（即 CUDA），因为找不到 CUDA 设备，所以报错。
    """
    print("正在加载模型...")
    read_model_path: Path = MODEL_PATH / 'multi_site_transformer_251204_V2.pth'
    model = TimeSeriesTransformer(CONFIG).to(CONFIG["device"])
    # 加载权重
    model.load_state_dict(torch.load(str(read_model_path)), map_location=torch.device('cpu'))
    model.eval()

    # 3. 进行推理
    preds_list = []
    targets_list = []
    inputs_dec_list = []  # 用于存储原始预报

    print("正在进行推理...")
    with torch.no_grad():
        for src, tgt, target in loader:
            src = src.to(CONFIG["device"])
            tgt = tgt.to(CONFIG["device"])

            # 模型预测 (订正后的值)
            output = model(src, tgt)

            preds_list.append(output.cpu().numpy())
            targets_list.append(target.numpy())
            inputs_dec_list.append(tgt.cpu().numpy())

    # 拼接所有 Batch
    # Shape: (样本数, 序列长度24, 特征数10)
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    orig_forecast = np.concatenate(inputs_dec_list, axis=0)

    # 4. 反归一化 (还原为 m/s)
    # 获取列索引
    df_cols = columns.tolist()
    # 目标特征 (实况) 的索引
    target_features = [f"{site}_real_{var}" for site in CONFIG["all_sites"] for var in ["u", "v"]]
    target_indices = [df_cols.index(col) for col in target_features]

    # 原始预报特征的索引 (注意：Decoder输入的就是预报)
    decoder_features = [f"{site}_forecast_{var}" for site in CONFIG["all_sites"] for var in ["u", "v"]]
    decoder_indices = [df_cols.index(col) for col in decoder_features]

    # 展平以便反归一化: (样本数*24, 特征数)
    N, T, F = preds.shape
    preds_flat = preds.reshape(-1, F)
    targets_flat = targets.reshape(-1, F)
    orig_forecast_flat = orig_forecast.reshape(-1, F)

    total_features = len(df_cols)

    # 还原数值
    print("正在反归一化数据...")
    # 还原预测值 (对应实况列)
    preds_real = inverse_transform_data(scaler, preds_flat, target_indices, total_features)
    # 还原真实值 (对应实况列)
    targets_real = inverse_transform_data(scaler, targets_flat, target_indices, total_features)
    # 还原原始预报 (对应预报列)
    # 注意：这里有个 trick，原始预报在 feature_df 里是在 forecast 列，但我们要把它和 real 列比较
    # 所以我们先还原 forecast 列的数值，然后直接拿来跟 real 比（物理单位都是 m/s）
    orig_forecast_real = inverse_transform_data(scaler, orig_forecast_flat, decoder_indices, total_features)

    # 5. 计算误差并绘图
    # 针对每个站点、每个变量(u, v)分别计算

    # 站点列表
    sites = CONFIG["all_sites"]

    # 为了方便，我们只取第一个站点 (例如 MF01002) 的 U 分量进行展示
    # 索引 0 对应第一个站点的 u
    site_idx = 0
    site_name = sites[0]
    var_name = "u"  # 或者 v
    feat_idx = 0  # 在 output_feature_num 中的相对索引

    # 提取单变量序列
    y_true = targets_real[:, feat_idx]
    y_pred = preds_real[:, feat_idx]  # 订正后
    y_orig = orig_forecast_real[:, feat_idx]  # 原始预报

    # --- 计算 RMSE ---
    rmse_orig = sqrt(mean_squared_error(y_true, y_orig))
    rmse_corr = sqrt(mean_squared_error(y_true, y_pred))
    improve = (rmse_orig - rmse_corr) / rmse_orig * 100

    print(f"\n[{site_name} {var_name}分量] 评估结果:")
    print(f"原始预报 RMSE: {rmse_orig:.4f} m/s")
    print(f"订正后   RMSE: {rmse_corr:.4f} m/s")
    print(f"提升率: {improve:.2f}%")

    # 6. 画图
    plt.figure(figsize=(12, 6))

    # 只画前 200 个小时的数据，避免太密看不清
    plot_len = 200
    x_axis = range(plot_len)

    plt.plot(x_axis, y_true[:plot_len], label='实况观测 (Actual)', color='black', linewidth=1.5)
    plt.plot(x_axis, y_orig[:plot_len], label=f'原始预报 (RMSE={rmse_orig:.2f})', color='blue', linestyle='--',
             alpha=0.7)
    plt.plot(x_axis, y_pred[:plot_len], label=f'订正后 (RMSE={rmse_corr:.2f})', color='red', linewidth=1.5)

    plt.title(f"{site_name} 风场({var_name}) 订正效果对比")
    plt.ylabel("风速 (m/s)")
    plt.xlabel("时间 (小时)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存或显示
    plt.savefig('evaluation_result.png')
    plt.show()

    # 7. 散点图 (Scatter Plot)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_orig, alpha=0.3, label='原始预报', s=10)
    plt.scatter(y_true, y_pred, alpha=0.3, label='订正后', s=10, color='red')

    # 画对角线
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()]),  # max of both axes
    ]
    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

    plt.xlabel("实况值")
    plt.ylabel("预测值")
    plt.title(f"{site_name} {var_name} 散点分布")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    evaluate()
