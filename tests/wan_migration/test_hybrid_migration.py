#!/usr/bin/env python3
"""
测试 Hybrid Wan Pipeline 迁移后的代码
验证所有组件是否正确导入和初始化
"""

import sys
import os
sys.path.insert(0, '/data/chenjiayu/minyu_lee/Hybrid-sd_wan')

print("=" * 60)
print("测试 Hybrid Wan Pipeline 迁移")
print("=" * 60)

# 测试1: 导入测试
print("\n测试1: 导入 HybridVideoInferencePipeline")
try:
    from compression.hybrid_sd.inference_pipeline import HybridVideoInferencePipeline
    print("✅ HybridVideoInferencePipeline 导入成功")
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试2: 检查 HybridWanPipeline 导入
print("\n测试2: 检查 HybridWanPipeline 导入")
try:
    from compression.hybrid_sd.diffusers.pipeline_wan import HybridWanPipeline
    print("✅ HybridWanPipeline 导入成功")
except Exception as e:
    print(f"❌ HybridWanPipeline 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: 创建 Pipeline 实例（不加载模型）
print("\n测试3: 创建 HybridVideoInferencePipeline 实例")

class MockArgs:
    def __init__(self):
        self.steps = [10, 15]  # 14B用10步，1.3B用15步
        self.enable_xformers_memory_efficient_attention = False
        self.use_custom_scheduler = False
        self.vae_device = None

model_path_14b = "/data/chenjiayu/models/Wan2.1-T2V-14B-Diffusers"
model_path_1_3b = "/data/chenjiayu/models/Wan2.1-T2V-1.3B-Diffusers"

# 检查模型路径
if not os.path.exists(model_path_14b):
    print(f"❌ 14B模型路径不存在: {model_path_14b}")
    sys.exit(1)
if not os.path.exists(model_path_1_3b):
    print(f"❌ 1.3B模型路径不存在: {model_path_1_3b}")
    sys.exit(1)

print(f"✅ 模型路径验证通过")
print(f"   - 14B: {model_path_14b}")
print(f"   - 1.3B: {model_path_1_3b}")

weight_folders = [model_path_14b, model_path_1_3b]
args = MockArgs()

try:
    pipeline = HybridVideoInferencePipeline(
        weight_folders=weight_folders,
        seed=42,
        device="cuda:0",
        args=args
    )
    print("✅ HybridVideoInferencePipeline 实例创建成功")
except Exception as e:
    print(f"❌ 实例创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: 加载模型组件
print("\n测试4: 加载模型组件（这将需要一些时间...）")
try:
    pipeline.set_pipe_and_generator()
    print("✅ 所有模型组件加载成功！")
    print(f"   - Text Encoder: UMT5")
    print(f"   - VAE: AutoencoderKLWan")
    print(f"   - Transformers: {len(weight_folders)} 个")
    print(f"   - Scheduler: {type(pipeline.pipe.scheduler).__name__}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有测试通过！Hybrid Wan Pipeline 迁移成功！")
print("=" * 60)
print("\n下一步: 可以运行完整的视频生成测试")
