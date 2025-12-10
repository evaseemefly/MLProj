import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt, atan2, degrees, pi

# 引入配置和模型
from train_transformer import CONFIG, load_and_prepare_data, create_samples, TimeSeriesTransformer, MultiSiteDataset
from torch.utils.data import DataLoader

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def inverse_transform_data(scaler, data_scaled, feature_indices, total_features):
    """辅助函数：反归一化"""
    N = data_scaled.shape[0]
    dummy = np.zeros((N, total_features))
    dummy[:, feature_indices] = data_scaled
    inversed_dummy = scaler.inverse_transform(dummy)
    return inversed_dummy[:, feature_indices]


def uv_to_speed_direction(u, v):
    """
    将 U, V 分量转换为风速和风向。
    风向：0-360度，气象定义（风的来向）。
    """
    # 1. 计算风速
    speed = np.sqrt(u ** 2 + v ** 2)

    # 2. 计算风向
    # Math angle (radians): 0 is East, 90 is North
    # degrees(atan2(v, u)) gives angle from x-axis (East)
    math_angle = np.degrees(np.arctan2(v, u))

    # Convert to Meteorological direction (0 is North, 90 is East, 180 is South, 270 is West)
    # Formula: wd = (270 - math_angle) % 360
    direction = (270 - math_angle) % 360

    return speed, direction


def calculate_direction_error(true_dir, pred_dir):
    """
    计算风向误差，处理 0/360 度的圆周问题。
    例如：真实359度，预测1度，误差应该是2度，而不是358度。
    """
    diff = np.abs(true_dir - pred_dir)
    # 取劣弧（最小夹角）
    error = np.minimum(diff, 360 - diff)
    return error


def evaluate_comprehensive():
    # 1. 准备数据
    print("正在加载数据...")
    merged_df = load_and_prepare_data(CONFIG)
    encoder_x, decoder_x, target_y, scaler, columns = create_samples(merged_df, CONFIG)

    # 划分验证集
    train_size = int(len(encoder_x) * 0.8)
    test_enc = encoder_x[train_size:]
    test_dec = decoder_x[train_size:]
    test_y = target_y[train_size:]

    dataset = MultiSiteDataset(test_enc, test_dec, test_y)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 2. 加载模型 (请确保路径正确)
    print("正在加载最佳模型...")
    model = TimeSeriesTransformer(CONFIG).to(CONFIG["device"])
    # 务必加载 best 模型
    model.load_state_dict(torch.load(r'H:/DATA/ML/MODEL/251127/multi_site_transformer_best.pth'))
    model.eval()

    # 3. 推理
    preds_list, targets_list, inputs_dec_list = [], [], []

    with torch.no_grad():
        for src, tgt, target in loader:
            src, tgt = src.to(CONFIG["device"]), tgt.to(CONFIG["device"])
            output = model(src, tgt)
            preds_list.append(output.cpu().numpy())
            targets_list.append(target.numpy())
            inputs_dec_list.append(tgt.cpu().numpy())

    # 拼接
    preds = np.concatenate(preds_list, axis=0)  # (样本数, 24, 特征数10)
    targets = np.concatenate(targets_list, axis=0)
    orig_forecast = np.concatenate(inputs_dec_list, axis=0)

    # 4. 反归一化
    df_cols = columns.tolist()
    total_features = len(df_cols)

    # 获取 U 和 V 的索引
    # 假设我们按站点遍历。MultiSiteDataset 输出的是所有站点的 [u, v]
    # 我们需要解析列名来找到成对的 u, v

    # 4. 反归一化 (保持不变，因为 scaler 需要 2D)
    N, T, F = preds.shape  # 关键：先记住 N (样本数) 和 T (预报步长，例如 72)
    preds_flat = preds.reshape(-1, F)
    targets_flat = targets.reshape(-1, F)
    orig_flat = orig_forecast.reshape(-1, F)

    target_indices = [df_cols.index(col) for col in
                      [f"{site}_real_{var}" for site in CONFIG["all_sites"] for var in ["u", "v"]]]
    decoder_indices = [df_cols.index(col) for col in
                       [f"{site}_forecast_{var}" for site in CONFIG["all_sites"] for var in ["u", "v"]]]

    preds_real = inverse_transform_data(scaler, preds_flat, target_indices, total_features)
    targets_real = inverse_transform_data(scaler, targets_flat, target_indices, total_features)
    orig_real = inverse_transform_data(scaler, orig_flat, decoder_indices, total_features)

    # 5. 综合评估 (针对第一个站点进行示例)
    # CONFIG["all_sites"] = ['MF01002', 'MF01004'...]
    # 每个站点占 2 列 (u, v)

    site_idx = 0
    site_name = CONFIG["all_sites"][site_idx]

    # 提取 U, V 分量
    # 预测值 —— predict ——预测
    pred_u = preds_real[:, site_idx * 2]
    """预测-u"""
    pred_v = preds_real[:, site_idx * 2 + 1]
    """预测-v"""
    # 真实值 —— true
    true_u = targets_real[:, site_idx * 2]
    """真实-u"""
    true_v = targets_real[:, site_idx * 2 + 1]
    """真实-v"""
    # 原始预报 —— origin
    orig_u = orig_real[:, site_idx * 2]
    """原始预报-u"""
    orig_v = orig_real[:, site_idx * 2 + 1]
    """原始预报-v"""

    # --- 核心：合成风速和风向 ---
    true_spd, true_dir = uv_to_speed_direction(true_u, true_v)

    pred_spd, pred_dir = uv_to_speed_direction(pred_u, pred_v)
    orig_spd, orig_dir = uv_to_speed_direction(orig_u, orig_v)
    """原始预报——speed"""

    # 6. 计算评估指标

    # A. 风速 RMSE
    rmse_spd_orig = sqrt(mean_squared_error(true_spd, orig_spd))
    rmse_spd_corr = sqrt(mean_squared_error(true_spd, pred_spd))

    # B. 风向 平均绝对误差 (MAE) - 需要处理环形误差
    mae_dir_orig = np.mean(calculate_direction_error(true_dir, orig_dir))
    mae_dir_corr = np.mean(calculate_direction_error(true_dir, pred_dir))

    # C. 矢量 RMSE (Vector RMSE) - 最硬核的指标
    # VRMSE = sqrt( mean( (u_pred - u_true)^2 + (v_pred - v_true)^2 ) )
    vrmse_orig = sqrt(np.mean((orig_u - true_u) ** 2 + (orig_v - true_v) ** 2))
    vrmse_corr = sqrt(np.mean((pred_u - true_u) ** 2 + (pred_v - true_v) ** 2))

    print(f"\n===== 站点 {site_name} 综合评估报告 =====")
    print(f"{'指标':<15} | {'原始预报':<10} | {'订正后':<10} | {'提升率':<10}")
    print("-" * 55)
    print(
        f"{'风速 RMSE (m/s)':<15} | {rmse_spd_orig:.4f}      | {rmse_spd_corr:.4f}      | {((rmse_spd_orig - rmse_spd_corr) / rmse_spd_orig):.2%}")
    print(
        f"{'风向 MAE (deg)':<15} | {mae_dir_orig:.4f}      | {mae_dir_corr:.4f}      | {((mae_dir_orig - mae_dir_corr) / mae_dir_orig):.2%}")
    print(
        f"{'矢量 VRMSE':<15} | {vrmse_orig:.4f}      | {vrmse_corr:.4f}      | {((vrmse_orig - vrmse_corr) / vrmse_orig):.2%}")

    # 7. 绘图 (修改部分)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))  # 移除 sharex=True，因为散点图的 x 轴含义变了

    plot_len = 240  # 画10天
    x = range(plot_len)

    # TODO:[?] 25-12-09 此处加载的 风速的长度为何为 13920？，我理解应该是预报的长度才对？是将多组数据拼接到了一起？
    # true_spd,orig_spd,pred_spd 长度均为 13920
    # 子图1: 风速时序对比 (保持不变)
    axes[0].plot(x, true_spd[:plot_len], 'k-', label='实况', linewidth=1.5)
    axes[0].plot(x, orig_spd[:plot_len], 'b--', label=f'原始预报 (RMSE={rmse_spd_orig:.2f})', alpha=0.6)
    axes[0].plot(x, pred_spd[:plot_len], 'r-', label=f'订正后 (RMSE={rmse_spd_corr:.2f})', linewidth=1.5)
    axes[0].set_ylabel('风速 (m/s)')
    axes[0].set_xlabel('时间 (小时)')
    axes[0].set_title(f'{site_name} 风速时序对比')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子图2: 风速散点对比图 (修改)
    # 绘制 "实况 vs 原始预报"
    axes[1].scatter(true_spd, orig_spd, alpha=0.3, s=10, c='blue', label='原始预报')
    # 绘制 "实况 vs 订正后"
    axes[1].scatter(true_spd, pred_spd, alpha=0.3, s=10, c='red', label='订正后')

    # 绘制 1:1 参考线
    max_val = max(np.max(true_spd), np.max(orig_spd), np.max(pred_spd))
    axes[1].plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, label='1:1 参考线')

    axes[1].set_xlabel('实况风速 (m/s)')
    axes[1].set_ylabel('预测风速 (m/s)')
    axes[1].set_title(f'{site_name} 风速散点对比 (Observed vs Predicted)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('comprehensive_evaluation.png')
    plt.show()


if __name__ == "__main__":
    evaluate_comprehensive()