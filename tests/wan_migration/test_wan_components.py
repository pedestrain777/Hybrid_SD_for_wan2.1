#!/usr/bin/env python3
"""
测试Wan模型组件加载
逐个测试 Text Encoder, Tokenizer, VAE, Transformer
"""

import sys
import torch

print("=" * 60)
print("测试Wan组件加载")
print("=" * 60)

# 模型路径
MODEL_PATH_14B = "/data/chenjiayu/models/Wan2.1-T2V-14B-Diffusers"
MODEL_PATH_1_3B = "/data/chenjiayu/models/Wan2.1-T2V-1.3B-Diffusers"

print(f"\n模型路径:")
print(f"  14B: {MODEL_PATH_14B}")
print(f"  1.3B: {MODEL_PATH_1_3B}")

# 检查路径是否存在
import os
if not os.path.exists(MODEL_PATH_14B):
    print(f"\n❌ 14B模型路径不存在: {MODEL_PATH_14B}")
    sys.exit(1)
if not os.path.exists(MODEL_PATH_1_3B):
    print(f"\n❌ 1.3B模型路径不存在: {MODEL_PATH_1_3B}")
    sys.exit(1)

print(f"\n✅ 模型路径都存在")

# 导入必要的库
from transformers import UMT5EncoderModel, AutoTokenizer
from diffusers import AutoencoderKLWan, WanTransformer3DModel

# 测试1: Text Encoder (UMT5)
print(f"\n" + "=" * 60)
print("测试1: UMT5 Text Encoder")
print("=" * 60)
try:
    print(f"加载 UMT5EncoderModel...")
    text_encoder = UMT5EncoderModel.from_pretrained(
        MODEL_PATH_14B,
        subfolder="text_encoder"
    )
    print(f"✅ UMT5 加载成功")
    print(f"   - 参数量: {sum(p.numel() for p in text_encoder.parameters()) / 1e9:.2f}B")
    print(f"   - 设备: {next(text_encoder.parameters()).device}")

    # 转移到GPU并设置为float16
    text_encoder = text_encoder.to("cuda:0", dtype=torch.float16)
    print(f"   - 已转移到 cuda:0 (float16)")

    # 清理显存
    del text_encoder
    torch.cuda.empty_cache()
    print(f"   - 已清理显存")

except Exception as e:
    print(f"❌ UMT5 加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试2: Tokenizer
print(f"\n" + "=" * 60)
print("测试2: AutoTokenizer")
print("=" * 60)
try:
    print(f"加载 AutoTokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH_14B,
        subfolder="tokenizer"
    )
    print(f"✅ Tokenizer 加载成功")
    print(f"   - 词汇表大小: {len(tokenizer)}")

    # 测试tokenize
    test_text = "A cat walking in the garden"
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"   - 测试文本: '{test_text}'")
    print(f"   - Token数量: {tokens['input_ids'].shape[1]}")

except Exception as e:
    print(f"❌ Tokenizer 加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: VAE
print(f"\n" + "=" * 60)
print("测试3: AutoencoderKLWan")
print("=" * 60)
try:
    print(f"加载 AutoencoderKLWan...")
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_PATH_14B,
        subfolder="vae"
    )
    print(f"✅ VAE 加载成功")
    print(f"   - 参数量: {sum(p.numel() for p in vae.parameters()) / 1e6:.2f}M")

    # 转移到GPU并设置为float16
    vae = vae.to("cuda:0", dtype=torch.float16)
    print(f"   - 已转移到 cuda:0 (float16)")

    # 启用优化
    vae.enable_slicing()
    vae.enable_tiling()
    print(f"   - 已启用 slicing 和 tiling")

    # 清理显存
    del vae
    torch.cuda.empty_cache()
    print(f"   - 已清理显存")

except Exception as e:
    print(f"❌ VAE 加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: Transformer (14B)
print(f"\n" + "=" * 60)
print("测试4: WanTransformer3DModel (14B)")
print("=" * 60)
try:
    print(f"加载 WanTransformer3DModel (14B)...")
    print(f"   使用 device_map='auto' 进行多GPU分配...")

    transformer_14b = WanTransformer3DModel.from_pretrained(
        MODEL_PATH_14B,
        subfolder="transformer",
        torch_dtype=torch.float16,
        device_map="auto"  # 自动多GPU分配
    )
    print(f"✅ Transformer 14B 加载成功")
    print(f"   - 参数量: {sum(p.numel() for p in transformer_14b.parameters()) / 1e9:.2f}B")

    # 检查设备分配
    devices = set()
    for name, param in transformer_14b.named_parameters():
        devices.add(str(param.device))
    print(f"   - 分配到的设备: {devices}")

    # 清理显存
    del transformer_14b
    torch.cuda.empty_cache()
    print(f"   - 已清理显存")

except Exception as e:
    print(f"❌ Transformer 14B 加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试5: Transformer (1.3B)
print(f"\n" + "=" * 60)
print("测试5: WanTransformer3DModel (1.3B)")
print("=" * 60)
try:
    print(f"加载 WanTransformer3DModel (1.3B)...")

    transformer_1_3b = WanTransformer3DModel.from_pretrained(
        MODEL_PATH_1_3B,
        subfolder="transformer",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print(f"✅ Transformer 1.3B 加载成功")
    print(f"   - 参数量: {sum(p.numel() for p in transformer_1_3b.parameters()) / 1e9:.2f}B")

    # 检查设备分配
    devices = set()
    for name, param in transformer_1_3b.named_parameters():
        devices.add(str(param.device))
    print(f"   - 分配到的设备: {devices}")

    # 清理显存
    del transformer_1_3b
    torch.cuda.empty_cache()
    print(f"   - 已清理显存")

except Exception as e:
    print(f"❌ Transformer 1.3B 加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有组件加载测试通过！")
print("=" * 60)
print("\n下一步: 运行 test_wan_single.py 测试单模型Pipeline")
