import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 版本: {torch.version.cuda}")
print(f"显卡是否可用: {torch.cuda.is_available()}")

# 关键：检查是否支持你的 sm_120 架构
# 如果这里报错或显示不支持，说明 Nightly 版还没跟上 5070
try:
    x = torch.ones(1).cuda()
    print("✅ GPU 计算测试成功！内核可用。")
except Exception as e:
    print(f"❌ GPU 测试失败: {e}")