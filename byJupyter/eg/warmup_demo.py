import numpy as np


def create_dataset_with_warmup(dataset, look_back=32, warmup_steps=10):
    """
    创建带有预热期的数据集
    :param dataset: 输入数据集
    :param look_back: 目标预测的序列长度
    :param warmup_steps: 预热步数
    :return: X (包含预热段的输入), Y (不含预热段的目标输出)
    """
    dataX, dataY = [], []
    # 总的输入长度 = 预热期 + 预测期
    # eg:6
    total_input_length = look_back + warmup_steps

    # 10-6+1=5
    for i in range(len(dataset) - total_input_length + 1):
        # 1. 提取输入X：包含预热段和预测段
        # 例如: warmup=10, look_back=32, X取第 i 到 i+42 个点
        a = dataset[i:(i + total_input_length), :]
        dataX.append(a)

        # 2. 提取目标Y：只包含我们关心的预测段
        # Y取第 i+10 到 i+42 个点，与X的后32个点对应
        b = dataset[(i + warmup_steps):(i + total_input_length), :]
        dataY.append(b)

    return np.array(dataX), np.array(dataY)


def main():
    # 1. 示例数据
    # 这是一个有10个时间点，1个特征的示例数据集
    # 它的形状是 (10, 1)
    sample_data = np.array([
        [10], [20], [30], [40], [50], [60], [70], [80], [90], [100]
    ])
    # 2. 【关键部分】明确定义并传入参数
    look_back_for_example = 4  # 目标预测序列的长度
    warmup_steps_for_example = 2  # 预热序列的长度
    # 3. 使用这些明确的参数来调用函数
    dataX_3d, dataY_3d = create_dataset_with_warmup(sample_data, look_back=look_back_for_example,
                                                    warmup_steps=warmup_steps_for_example)

    # 3. 【关键步骤】将三维数组转换为二维数组以便阅读
    # np.squeeze() 会移除所有长度为1的维度。 (5, 6, 1) -> (5, 6)
    dataX_2d = np.squeeze(dataX_3d)
    dataY_2d = np.squeeze(dataY_3d)
    pass


if __name__ == '__main__':
    main()
