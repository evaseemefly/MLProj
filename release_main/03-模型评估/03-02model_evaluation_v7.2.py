import pandas as pd
import numpy as np
import pathlib
import math
import joblib
import arrow
from tensorflow.keras.models import load_model
from typing import Any


# 您原来的辅助函数，保持不变
def load_customer_model(model_path: str) -> Any:
    if pathlib.Path(model_path).exists():
        loaded_model = load_model(model_path)
        print("模型结构概览:")
        loaded_model.summary()
        return loaded_model
    print(f"错误：模型文件不存在于 {model_path}")
    return None


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> pd.Series:
    assert y_true.shape == y_pred.shape, "输入数组的形状必须相同"
    squared_errors = (y_true - y_pred) ** 2
    # 使用 np.nanmean 忽略 NaN 值，更稳健
    mse_per_row = np.nanmean(squared_errors, axis=1)
    rmse_per_row = np.sqrt(mse_per_row)
    return rmse_per_row


# ==============================================================================
#  核心：为残差模型设计的全新评估函数
# ==============================================================================
def evaluate_residual_model(model_path: str,
                            forecast_data_path: str,
                            real_data_path: str,
                            scaler_forecast_path: str,
                            scaler_residual_path: str,  # TODO: 关键！为残差模型新增一个 scaler_residual_path
                            out_put_rmse_path: str,
                            dataset_count: int = 64,
                            warmup_count: int = 0):
    """
    专门用于评估“预测残差”模型的函数。

    Args:
        model_path (str): 训练好的模型路径 (.h5)。
        forecast_data_path (str): 原始预报数据路径 (.csv)。
        real_data_path (str): 原始实况数据路径 (.csv)。
        scaler_forecast_path (str): 用于归一化输入(预报)的标量器 (.sav)。
        scaler_residual_path (str): 用于归一化目标(残差)的标量器 (.sav)。
        out_put_rmse_path (str): RMSE 结果的输出目录。
        dataset_count (int): 使用的数据行数。
        warmup_count (int): 需要从结果中剔除的预热时长（小时数）。
    """
    # --- 步骤 1: 加载测试所需的原始数据 ---
    df_forecast = pd.read_csv(forecast_data_path, encoding='utf-8', index_col=0).iloc[:dataset_count, :]
    df_realdata = pd.read_csv(real_data_path, encoding='utf-8', index_col=0).iloc[:dataset_count, :]

    # 按照 train_test_split 的逻辑，获取测试集部分（后20%）
    split_count = math.ceil(df_forecast.shape[1] * 0.2)
    test_forecast_orig = df_forecast[df_forecast.columns[-split_count:]]
    test_real_orig = df_realdata[df_realdata.columns[-split_count:]]

    print(f"使用后 {split_count} 个样本作为测试集。")
    print(f"测试集原始预报数据形状: {test_forecast_orig.shape}")
    print(f"测试集原始实况数据形状: {test_real_orig.shape}")

    # --- 步骤 2: 准备模型输入 (归一化的预报数据) ---
    rows, cols = test_forecast_orig.shape
    # 将原始预报数据（测试集）转换为模型输入格式
    X_test_orig = test_forecast_orig.values.T.reshape(cols, rows, 1)

    # 加载预报数据的归一化器
    scaler_forecast = joblib.load(scaler_forecast_path)

    # 归一化模型输入
    X_test_scaled = scaler_forecast.transform(X_test_orig.reshape(-1, 1))
    X_test_scaled = X_test_scaled.reshape(X_test_orig.shape)

    # --- 步骤 3: 加载模型和所有必需的归一化器 ---
    model = load_customer_model(model_path)
    if model is None:
        return

    # 加载残差的归一化器，这是反向还原的关键！
    scaler_residual = joblib.load(scaler_residual_path)

    # --- 步骤 4: 预测归一化的残差 ---
    # 使用处理好的、归一化后的 X_test_scaled 进行预测
    predicted_residual_scaled = model.predict(X_test_scaled)  # shape: (样本数, 时间步, 1)

    # --- 步骤 5: 将预测的残差反归一化 ---
    # 将3D的预测结果拍扁成2D以进行反归一化
    pred_res_scaled_flat = predicted_residual_scaled.reshape(-1, 1)
    # 使用残差的标量器进行反归一化
    pred_res_unscaled_flat = scaler_residual.inverse_transform(pred_res_scaled_flat)
    # 恢复为3D形状，并去掉最后一个维度 (样本数, 时间步)
    predicted_residual_unscaled = pred_res_unscaled_flat.reshape(predicted_residual_scaled.shape).squeeze(axis=-1)
    # 转置以匹配原始数据格式 (时间步, 样本数)
    predicted_residual_unscaled = predicted_residual_unscaled.T
    print(f"反归一化后的预测残差形状: {predicted_residual_unscaled.shape}")

    # --- 步骤 6: 计算修正后的预报 (核心步骤) ---
    # 修正预报 = 原始预报 + 预测的残差
    # 注意：确保两者都是numpy array且形状一致
    corrected_forecast = test_forecast_orig.to_numpy() + predicted_residual_unscaled
    print(f"修正后的预报形状: {corrected_forecast.shape}")

    # --- 步骤 7: 剔除预热数据 ---
    # 在计算RMSE之前，从修正后的预报和真实实况中剔除预热部分
    final_corrected_forecast = corrected_forecast[warmup_count:, :]
    final_real_data = test_real_orig.to_numpy()[warmup_count:, :]
    print(f"剔除 {warmup_count} 小时预热数据后，最终评估数据形状: {final_corrected_forecast.shape}")

    # --- 步骤 8: 计算修正后预报的RMSE ---
    rmse_series = compute_rmse(final_real_data, final_corrected_forecast)

    # --- 步骤 9: 存储RMSE结果 ---
    now_arrow: arrow.Arrow = arrow.now()
    date_str: str = now_arrow.date().isoformat()
    rmse_file_name: str = f'rmse_residual_model_{date_str}.csv'
    rmse_full_path: str = pathlib.Path(out_put_rmse_path) / rmse_file_name
    rmse_df = pd.DataFrame(rmse_series, columns=['RMSE'])
    # 如果需要，可以为索引添加有意义的名称
    rmse_df.index = [f't+{i + 1 + warmup_count}h' for i in range(len(rmse_df))]
    rmse_df.to_csv(rmse_full_path)
    print(f'[-] 输出的RMSE存储路径为: {rmse_full_path}')


def main():
    model_path: str = r'Z:\02TRAINNING_MODEL\fit_model_v7.2_250714.h5'
    forecast_path: str = r'Z:\01TRAINNING_DATA\250713\2024_warmup_dataset_forecast_aligned.csv'  # 注意，这里的文件名我帮你修正了
    realdata_path: str = r'Z:\01TRAINNING_DATA\250713\2024_warmup_dataset_realdata.csv'

    # 定义三个归一化器的路径
    forecast_scaler_path: str = r'Z:\01TRAINNING_DATA\scaler\250714\scaler_forecast_2.sav'
    # ！！！重要：您需要提供在训练时保存的残差归一化器路径！！！
    # 在您的训练代码中，它应该是原来保存 `scaler_y` 的那个文件
    residual_scaler_path: str = r'Z:\01TRAINNING_DATA\scaler\250714\scaler_realdata_2.sav'  # 假设这就是残差标量器

    out_put_rmse_path: str = r'Z:\04TRAINNING_EVALUATION_PIC\MODEL_V7\7.2'

    warmup_count = 4

    evaluate_residual_model(
        model_path=model_path,
        forecast_data_path=forecast_path,
        real_data_path=realdata_path,
        scaler_forecast_path=forecast_scaler_path,
        scaler_residual_path=residual_scaler_path,  # 传递残差标量器路径
        out_put_rmse_path=out_put_rmse_path,
        dataset_count=64,
        warmup_count=warmup_count
    )


if __name__ == '__main__':
    main()
