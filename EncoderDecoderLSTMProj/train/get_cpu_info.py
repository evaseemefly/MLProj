import torch
print("-" * 30)
print(f"1. PyTorch 版本: {torch.__version__}")
print(f"2. CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"3. 显卡名称: {torch.cuda.get_device_name(0)}")
    print(f"4. PyTorch 编译时的 CUDA 版本: {torch.version.cuda}")
    # 检查 GPU 能力 (Compute Capability)
    cc = torch.cuda.get_device_capability(0)
    print(f"5. 显卡计算能力 (Compute Capability): {cc[0]}.{cc[1]}")
else:
    print("未检测到可用的 GPU 或 PyTorch 为 CPU 版本。")
print("-" * 30)

"""
    1. PyTorch 版本: 2.5.1+cu121
    2. CUDA 是否可用: True
    3. 显卡名称: NVIDIA GeForce RTX 5070
    4. PyTorch 编译时的 CUDA 版本: 12.1
    5. 显卡计算能力 (Compute Capability): 12.0
    ------------------------------
    D:\09-Setups\ANACONDA\envs\py39\lib\site-packages\torch\cuda\__init__.py:235: UserWarning: 
    NVIDIA GeForce RTX 5070 with CUDA capability sm_120 is not compatible with the current PyTorch installation.
    The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_61 sm_70 sm_75 sm_80 sm_86 sm_90.
    If you want to use the NVIDIA GeForce RTX 5070 GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/
    
      warnings.warn(
"""