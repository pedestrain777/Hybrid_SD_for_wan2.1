#!/usr/bin/env python3
"""
测试环境是否支持Wan系列模型
检查版本和组件导入
"""

import sys

print("=" * 60)
print("测试环境：检查版本和组件导入")
print("=" * 60)

# 1. 检查Python版本
print(f"\n1. Python版本:")
print(f"   {sys.version}")

# 2. 检查核心库版本
print(f"\n2. 核心库版本:")
try:
    import diffusers
    print(f"   ✅ diffusers: {diffusers.__version__}")

    # 检查版本是否满足要求
    from packaging import version
    if version.parse(diffusers.__version__) >= version.parse("0.29.0"):
        print(f"      ✅ 版本满足要求 (>=0.29.0)")
    else:
        print(f"      ⚠️  版本过低，需要 >=0.29.0")
except ImportError as e:
    print(f"   ❌ diffusers 未安装: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"   ✅ transformers: {transformers.__version__}")

    if version.parse(transformers.__version__) >= version.parse("4.37.0"):
        print(f"      ✅ 版本满足要求 (>=4.37.0)")
    else:
        print(f"      ⚠️  版本过低，需要 >=4.37.0")
except ImportError as e:
    print(f"   ❌ transformers 未安装: {e}")
    sys.exit(1)

try:
    import torch
    print(f"   ✅ torch: {torch.__version__}")

    if version.parse(torch.__version__.split('+')[0]) >= version.parse("2.1.0"):
        print(f"      ✅ 版本满足要求 (>=2.1.0)")
    else:
        print(f"      ⚠️  版本过低，需要 >=2.1.0")
except ImportError as e:
    print(f"   ❌ torch 未安装: {e}")
    sys.exit(1)

# 3. 检查CUDA
print(f"\n3. CUDA支持:")
if torch.cuda.is_available():
    print(f"   ✅ CUDA 可用")
    print(f"   - CUDA版本: {torch.version.cuda}")
    print(f"   - GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"     显存: {mem_total:.1f} GB")
else:
    print(f"   ❌ CUDA 不可用")

# 4. 测试Wan组件导入
print(f"\n4. 测试Wan组件导入:")

try:
    from diffusers import WanPipeline
    print(f"   ✅ WanPipeline 导入成功")
except ImportError as e:
    print(f"   ❌ WanPipeline 导入失败: {e}")
    print(f"      可能原因: diffusers版本过低 (<0.29.0)")
    sys.exit(1)

try:
    from diffusers import WanTransformer3DModel
    print(f"   ✅ WanTransformer3DModel 导入成功")
except ImportError as e:
    print(f"   ❌ WanTransformer3DModel 导入失败: {e}")
    sys.exit(1)

try:
    from diffusers import AutoencoderKLWan
    print(f"   ✅ AutoencoderKLWan 导入成功")
except ImportError as e:
    print(f"   ❌ AutoencoderKLWan 导入失败: {e}")
    sys.exit(1)

try:
    from diffusers import FlowMatchEulerDiscreteScheduler
    print(f"   ✅ FlowMatchEulerDiscreteScheduler 导入成功")
except ImportError as e:
    print(f"   ❌ FlowMatchEulerDiscreteScheduler 导入失败: {e}")
    sys.exit(1)

try:
    from transformers import UMT5EncoderModel
    print(f"   ✅ UMT5EncoderModel 导入成功")
except ImportError as e:
    print(f"   ❌ UMT5EncoderModel 导入失败: {e}")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
    print(f"   ✅ AutoTokenizer 导入成功")
except ImportError as e:
    print(f"   ❌ AutoTokenizer 导入失败: {e}")
    sys.exit(1)

# 5. 检查WanPipeline参数
print(f"\n5. 检查WanPipeline参数支持:")
import inspect
sig = inspect.signature(WanPipeline.__init__)
params = list(sig.parameters.keys())

required_params = ['transformer_2', 'boundary_ratio', 'expand_timesteps']
for param in required_params:
    if param in params:
        print(f"   ✅ {param} 参数存在")
    else:
        print(f"   ❌ {param} 参数不存在")

# 6. 检查FlowMatchEulerDiscreteScheduler.step参数
print(f"\n6. 检查FlowMatchEulerDiscreteScheduler.step参数:")
sig = inspect.signature(FlowMatchEulerDiscreteScheduler.step)
params = list(sig.parameters.keys())

required_params = ['s_churn', 's_noise', 's_tmin', 's_tmax']
for param in required_params:
    if param in params:
        print(f"   ✅ {param} 参数存在")
    else:
        print(f"   ❌ {param} 参数不存在")

print("\n" + "=" * 60)
print("✅ 环境检查完成！所有组件都可以正常导入")
print("=" * 60)
print("\n下一步: 运行 test_wan_components.py 测试组件加载")
