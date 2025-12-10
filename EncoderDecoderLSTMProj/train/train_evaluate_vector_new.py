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

    preds_real_flat = inverse_transform_data(scaler, preds_flat, target_indices, total_features)
    targets_real_flat = inverse_transform_data(scaler, targets_flat, target_indices, total_features)
    orig_real_flat = inverse_transform_data(scaler, orig_flat, decoder_indices, total_features)  # 原始预报反归一化

    # ================= 关键修改点开始 =================

    # 5. 还原维度 (Reshape Back)
    # 假设每个站点有 2 个特征 (u, v)，所以最后一维通常是 2 * 站点数
    # 但我们这里已经筛选出了 target_indices，所以特征数是 len(target_indices)
    n_vars = len(target_indices)

    preds_real_3d = preds_real_flat.reshape(N, T, n_vars)
    targets_real_3d = targets_real_flat.reshape(N, T, n_vars)
    orig_real_3d = orig_real_flat.reshape(N, T, n_vars)  # 原始预报还原 3D

    # 针对第一个站点进行分析 (假设 site_idx=0, u在0列, v在1列)
    site_idx = 0
    site_name = CONFIG["all_sites"][site_idx]

    # 提取 U, V 分量 (维度: [样本数, 预报步长])
    # 订正后
    pred_u = preds_real_3d[:, :, site_idx * 2]
    pred_v = preds_real_3d[:, :, site_idx * 2 + 1]
    # 实况
    true_u = targets_real_3d[:, :, site_idx * 2]
    true_v = targets_real_3d[:, :, site_idx * 2 + 1]
    # 原始预报
    orig_u = orig_real_3d[:, :, site_idx * 2]
    orig_v = orig_real_3d[:, :, site_idx * 2 + 1]

    # 合成风速
    pred_spd, _ = uv_to_speed_direction(pred_u, pred_v)
    true_spd, _ = uv_to_speed_direction(true_u, true_v)
    orig_spd, _ = uv_to_speed_direction(orig_u, orig_v)

    # 6. 按预报时效计算 RMSE
    pred_rmse_list = []
    orig_rmse_list = []
    steps = range(T)

    for t in steps:
        # 订正后 RMSE
        rmse_pred = sqrt(mean_squared_error(true_spd[:, t], pred_spd[:, t]))
        pred_rmse_list.append(rmse_pred)

        # 原始预报 RMSE (新增)
        rmse_orig = sqrt(mean_squared_error(true_spd[:, t], orig_spd[:, t]))
        orig_rmse_list.append(rmse_orig)

    # 打印对比结果
    print(f"\n===== 站点 {site_name} RMSE 对比 (部分时效) =====")
    print(f"{'时效':<5} | {'原始 RMSE':<10} | {'订正 RMSE':<10} | {'提升':<10}")
    check_points = [0, 3,6,9,12,15,18,21, 23, T - 1]  # 检查第1, 12, 24, 最后一个小时
    for t in check_points:
        if t < T:
            imp = (orig_rmse_list[t] - pred_rmse_list[t]) / orig_rmse_list[t]
            print(f"H+{t + 1:<3} | {orig_rmse_list[t]:.4f}     | {pred_rmse_list[t]:.4f}     | {imp:.2%}")

    # 7. 绘图：RMSE 对比
    plt.figure(figsize=(10, 6))

    # 绘制原始预报 RMSE
    plt.plot(range(1, T + 1), orig_rmse_list, 'b--o', markersize=4, label='原始预报 RMSE', alpha=0.7)
    # 绘制订正后 RMSE
    plt.plot(range(1, T + 1), pred_rmse_list, 'r-o', markersize=4, label='订正后 RMSE', linewidth=2)

    plt.title(f'{site_name} 风速预报误差对比 (RMSE vs Lead Time)')
    plt.xlabel('预报时效 (小时)')
    plt.ylabel('RMSE (m/s)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig('rmse_comparison_lead_time.png')
    plt.show()


if __name__ == "__main__":
    evaluate_comprehensive()
