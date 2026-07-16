#!/usr/bin/env python3
"""
测试CogVideoX-5B和CogVideoX-2B的transformer输入输出形状
使用实际的pipeline来确保参数正确
"""

import json
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_config_compatibility():
    """分析配置兼容性"""
    print("="*60)
    print("配置兼容性分析")
    print("="*60)
    
    model_path_5b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-5B"
    model_path_2b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-2B"
    
    # 加载配置
    config_5b_transformer = json.load(open(f"{model_path_5b}/transformer/config.json"))
    config_2b_transformer = json.load(open(f"{model_path_2b}/transformer/config.json"))
    config_5b_vae = json.load(open(f"{model_path_5b}/vae/config.json"))
    config_2b_vae = json.load(open(f"{model_path_2b}/vae/config.json"))
    
    print("\n关键参数对比:")
    print(f"{'参数':<40} | {'5B':<20} | {'2B':<20} | {'兼容'}")
    print("-" * 90)
    
    # 检查关键参数
    key_params = {
        'transformer': [
            ('in_channels', 'Transformer输入通道'),
            ('out_channels', 'Transformer输出通道'),
            ('sample_frames', '帧数'),
            ('sample_height', 'Latent高度'),
            ('sample_width', 'Latent宽度'),
            ('text_embed_dim', '文本嵌入维度'),
            ('time_embed_dim', '时间嵌入维度'),
            ('temporal_compression_ratio', '时间压缩比'),
            ('patch_size', 'Patch大小'),
        ],
        'vae': [
            ('latent_channels', 'VAE潜在通道'),
            ('sample_height', '视频高度'),
            ('sample_width', '视频宽度'),
            ('temporal_compression_ratio', '时间压缩比'),
        ]
    }
    
    all_compatible = True
    
    # Transformer参数
    print("\n【Transformer参数】")
    for param, desc in key_params['transformer']:
        val_5b = config_5b_transformer.get(param)
        val_2b = config_2b_transformer.get(param)
        compatible = val_5b == val_2b
        if not compatible:
            all_compatible = False
        status = "✅" if compatible else "❌"
        print(f"{desc:<40} | {str(val_5b):<20} | {str(val_2b):<20} | {status}")
    
    # VAE参数
    print("\n【VAE参数】")
    for param, desc in key_params['vae']:
        val_5b = config_5b_vae.get(param)
        val_2b = config_2b_vae.get(param)
        compatible = val_5b == val_2b
        if not compatible:
            all_compatible = False
        status = "✅" if compatible else "❌"
        print(f"{desc:<40} | {str(val_5b):<20} | {str(val_2b):<20} | {status}")
    
    # 显示不同但可能不关键的参数
    print("\n【其他参数差异】")
    diff_params = [
        ('num_attention_heads', '注意力头数'),
        ('num_layers', '层数'),
        ('use_rotary_positional_embeddings', '使用Rotary位置编码'),
        ('_diffusers_version', 'Diffusers版本'),
    ]
    
    for param, desc in diff_params:
        val_5b = config_5b_transformer.get(param)
        val_2b = config_2b_transformer.get(param)
        if val_5b != val_2b:
            print(f"  ⚠️  {desc}: 5B={val_5b}, 2B={val_2b}")
    
    # 计算latent形状
    print("\n【Latent形状计算】")
    patch_size = config_5b_transformer['patch_size']
    h_latent = config_5b_transformer['sample_height']  # 60
    w_latent = config_5b_transformer['sample_width']   # 90
    t_latent = config_5b_transformer['sample_frames']  # 49
    c_latent = config_5b_transformer['in_channels']    # 16
    
    h_patches = h_latent // patch_size  # 30
    w_patches = w_latent // patch_size  # 45
    spatial_seq_len = h_patches * w_patches  # 1350
    total_image_seq_len = t_latent * spatial_seq_len  # 66150
    
    print(f"  Latent形状 (B, T, C, H, W): (B, {t_latent}, {c_latent}, {h_latent}, {w_latent})")
    print(f"  Patch后空间序列长度: {h_patches} * {w_patches} = {spatial_seq_len}")
    print(f"  Patch后总图像序列长度: {t_latent} * {spatial_seq_len} = {total_image_seq_len}")
    print(f"  文本序列长度 (max): {config_5b_transformer.get('max_text_seq_length', 'N/A')}")
    
    # 结论
    print("\n" + "="*60)
    print("兼容性结论")
    print("="*60)
    
    if all_compatible:
        print("\n✅✅✅ 关键参数完全兼容！")
        print("   - Latent空间维度完全相同")
        print("   - 输入输出通道数相同")
        print("   - 理论上可以无缝切换transformer模型")
        print("\n⚠️  注意事项:")
        print("   - use_rotary_positional_embeddings不同（5B使用，2B不使用）")
        print("   - 在切换时需要根据模型类型处理位置编码")
        print("   - num_attention_heads和num_layers不同，但不影响latent兼容性")
    else:
        print("\n❌ 关键参数不兼容")
        print("   需要解决兼容性问题才能实现模型切换")
    
    return all_compatible

def test_with_pipeline():
    """使用pipeline测试（如果可能）"""
    print("\n" + "="*60)
    print("使用Pipeline测试（可选）")
    print("="*60)
    
    try:
        from diffusers import CogVideoXPipeline
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n使用设备: {device}")
        
        # 测试5B模型
        print("\n测试5B模型pipeline...")
        try:
            pipeline_5b = CogVideoXPipeline.from_pretrained(
                "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-5B",
                torch_dtype=torch.float16
            ).to(device)
            print("✅ 5B pipeline加载成功")
        except Exception as e:
            print(f"⚠️  5B pipeline加载失败: {e}")
            return
        
        # 测试2B模型
        print("\n测试2B模型pipeline...")
        try:
            pipeline_2b = CogVideoXPipeline.from_pretrained(
                "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-2B",
                torch_dtype=torch.float16
            ).to(device)
            print("✅ 2B pipeline加载成功")
        except Exception as e:
            print(f"⚠️  2B pipeline加载失败: {e}")
            return
        
        # 检查transformer配置
        print("\nTransformer配置对比:")
        config_5b = pipeline_5b.transformer.config
        config_2b = pipeline_2b.transformer.config
        
        print(f"  5B use_rotary: {config_5b.use_rotary_positional_embeddings}")
        print(f"  2B use_rotary: {config_2b.use_rotary_positional_embeddings}")
        print(f"  5B in_channels: {config_5b.in_channels}")
        print(f"  2B in_channels: {config_2b.in_channels}")
        print(f"  5B out_channels: {config_5b.out_channels}")
        print(f"  2B out_channels: {config_2b.out_channels}")
        
        # 测试latent形状
        print("\n测试latent形状兼容性...")
        batch_size = 1
        num_frames = config_5b.sample_frames
        latent_channels = config_5b.in_channels
        latent_height = config_5b.sample_height
        latent_width = config_5b.sample_width
        
        test_latent = torch.randn(
            batch_size, num_frames, latent_channels, latent_height, latent_width,
            device=device, dtype=torch.float16
        )
        
        print(f"  测试latent形状: {test_latent.shape}")
        
        # 验证两个模型的latent形状要求是否相同
        if (config_5b.in_channels == config_2b.in_channels and
            config_5b.sample_frames == config_2b.sample_frames and
            config_5b.sample_height == config_2b.sample_height and
            config_5b.sample_width == config_2b.sample_width):
            print("✅ Latent形状要求相同")
        else:
            print("❌ Latent形状要求不同")
        
    except ImportError as e:
        print(f"⚠️  无法导入diffusers: {e}")
        print("   跳过pipeline测试")
    except Exception as e:
        print(f"⚠️  Pipeline测试失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("="*60)
    print("CogVideoX Transformer 形状兼容性测试")
    print("="*60)
    
    # 1. 配置兼容性分析
    compatible = analyze_config_compatibility()
    
    # 2. Pipeline测试（如果可能）
    test_with_pipeline()
    
    # 3. 最终建议
    print("\n" + "="*60)
    print("实现建议")
    print("="*60)
    
    if compatible:
        print("\n✅ 可以实现的方案:")
        print("   1. 两个模型的latent空间完全兼容")
        print("   2. 可以在denoising loop中动态切换transformer")
        print("   3. 需要根据模型类型处理use_rotary_positional_embeddings")
        print("   4. 可以共享VAE和Text Encoder")
        print("\n⚠️  注意事项:")
        print("   1. 5B模型使用rotary positional embeddings，2B不使用")
        print("   2. 在切换时需要正确设置image_rotary_emb参数")
        print("   3. num_attention_heads和num_layers不同，但不影响latent兼容性")
        print("   4. 需要在实际推理中测试跨模型切换的稳定性")
    else:
        print("\n❌ 需要解决的问题:")
        print("   1. 解决latent维度不兼容的问题")
        print("   2. 或者使用latent转换层")

if __name__ == "__main__":
    main()

