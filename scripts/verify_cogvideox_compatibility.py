#!/usr/bin/env python3
"""
验证CogVideoX-5B和CogVideoX-2B的transformer兼容性
"""

import json
import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)

def compare_configs(config_5b, config_2b, config_name):
    """比较两个配置"""
    print(f"\n{'='*60}")
    print(f"比较 {config_name} 配置")
    print(f"{'='*60}")
    
    # 关键参数列表
    key_params = [
        'in_channels', 'out_channels', 
        'sample_frames', 'sample_height', 'sample_width',
        'text_embed_dim', 'time_embed_dim',
        'temporal_compression_ratio',
        'use_rotary_positional_embeddings'
    ]
    
    compatible = True
    differences = []
    
    for param in key_params:
        val_5b = config_5b.get(param, 'N/A')
        val_2b = config_2b.get(param, 'N/A')
        
        if val_5b == val_2b:
            status = "✅"
        else:
            status = "❌"
            compatible = False
            differences.append((param, val_5b, val_2b))
        
        print(f"{status} {param:40s} | 5B: {str(val_5b):20s} | 2B: {str(val_2b):20s}")
    
    # 显示其他差异
    other_params_5b = set(config_5b.keys()) - set(key_params)
    other_params_2b = set(config_2b.keys()) - set(key_params)
    
    if other_params_5b or other_params_2b:
        print(f"\n其他参数差异:")
        all_params = other_params_5b.union(other_params_2b)
        for param in sorted(all_params):
            val_5b = config_5b.get(param, 'N/A')
            val_2b = config_2b.get(param, 'N/A')
            if val_5b != val_2b:
                print(f"  ⚠️  {param:40s} | 5B: {str(val_5b):20s} | 2B: {str(val_2b):20s}")
    
    return compatible, differences

def test_transformer_shapes():
    """测试transformer的输入输出形状"""
    print(f"\n{'='*60}")
    print("测试Transformer输入输出形状")
    print(f"{'='*60}")
    
    try:
        from diffusers import CogVideoXTransformer3DModel
        
        model_path_5b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-5B"
        model_path_2b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-2B"
        
        print("\n加载模型...")
        print(f"5B模型路径: {model_path_5b}")
        print(f"2B模型路径: {model_path_2b}")
        
        # 加载transformer（不加载权重，只检查结构）
        print("\n加载5B transformer配置...")
        config_5b_path = f"{model_path_5b}/transformer/config.json"
        config_5b = load_config(config_5b_path)
        
        print("加载2B transformer配置...")
        config_2b_path = f"{model_path_2b}/transformer/config.json"
        config_2b = load_config(config_2b_path)
        
        # 创建测试输入
        # Latent形状: (B, T, C, H, W)
        # 根据config: sample_frames=49, sample_height=60, sample_width=90, in_channels=16
        batch_size = 1
        num_frames = config_5b['sample_frames']  # 49
        latent_channels = config_5b['in_channels']  # 16
        latent_height = config_5b['sample_height']  # 60
        latent_width = config_5b['sample_width']  # 90
        
        print(f"\n测试输入形状:")
        print(f"  Batch size: {batch_size}")
        print(f"  Frames (T): {num_frames}")
        print(f"  Channels (C): {latent_channels}")
        print(f"  Height (H): {latent_height}")
        print(f"  Width (W): {latent_width}")
        print(f"  Latent形状: ({batch_size}, {num_frames}, {latent_channels}, {latent_height}, {latent_width})")
        
        # 创建模拟输入
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16
        
        print(f"\n设备: {device}, 数据类型: {dtype}")
        
        # 创建latent输入 (B, T, C, H, W)
        latent_input = torch.randn(
            batch_size, num_frames, latent_channels, latent_height, latent_width,
            device=device, dtype=dtype
        )
        
        # 创建timestep
        timestep = torch.tensor([500], device=device, dtype=torch.long)
        
        # 创建encoder_hidden_states (B, seq_len, text_embed_dim)
        text_embed_dim = config_5b['text_embed_dim']  # 4096
        seq_len = 77  # 常见的文本序列长度
        encoder_hidden_states = torch.randn(
            batch_size, seq_len, text_embed_dim,
            device=device, dtype=dtype
        )
        
        print(f"\n输入形状:")
        print(f"  latent_input: {latent_input.shape}")
        print(f"  timestep: {timestep.shape}")
        print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
        
        # 尝试加载模型（如果可能）
        print(f"\n{'='*60}")
        print("尝试加载模型测试...")
        print(f"{'='*60}")
        
        try:
            print("\n加载5B transformer...")
            transformer_5b = CogVideoXTransformer3DModel.from_pretrained(
                f"{model_path_5b}/transformer",
                torch_dtype=dtype
            ).to(device)
            transformer_5b.eval()
            
            print("加载2B transformer...")
            transformer_2b = CogVideoXTransformer3DModel.from_pretrained(
                f"{model_path_2b}/transformer",
                torch_dtype=dtype
            ).to(device)
            transformer_2b.eval()
            
            print("\n测试5B transformer前向传播...")
            with torch.no_grad():
                output_5b = transformer_5b(
                    hidden_states=latent_input,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    return_dict=False
                )[0]
            
            print(f"✅ 5B输出形状: {output_5b.shape}")
            
            print("\n测试2B transformer前向传播...")
            with torch.no_grad():
                output_2b = transformer_2b(
                    hidden_states=latent_input,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    return_dict=False
                )[0]
            
            print(f"✅ 2B输出形状: {output_2b.shape}")
            
            # 检查输出形状是否相同
            if output_5b.shape == output_2b.shape:
                print(f"\n✅✅✅ 输出形状兼容！")
                print(f"   两个模型的输出形状相同: {output_5b.shape}")
                return True
            else:
                print(f"\n❌❌❌ 输出形状不兼容！")
                print(f"   5B输出形状: {output_5b.shape}")
                print(f"   2B输出形状: {output_2b.shape}")
                return False
                
        except Exception as e:
            print(f"\n⚠️  无法加载模型进行测试: {e}")
            print("   但可以从配置文件中分析兼容性")
            return None
            
    except ImportError as e:
        print(f"\n⚠️  无法导入diffusers: {e}")
        print("   请安装diffusers库: pip install diffusers")
        return None
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    print("="*60)
    print("CogVideoX-5B 和 CogVideoX-2B 兼容性验证")
    print("="*60)
    
    model_path_5b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-5B"
    model_path_2b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-2B"
    
    # 检查模型路径
    if not os.path.exists(model_path_5b):
        print(f"❌ 5B模型路径不存在: {model_path_5b}")
        return
    if not os.path.exists(model_path_2b):
        print(f"❌ 2B模型路径不存在: {model_path_2b}")
        return
    
    # 1. 比较Transformer配置
    print("\n" + "="*60)
    print("1. 比较Transformer配置")
    print("="*60)
    
    config_5b_transformer = load_config(f"{model_path_5b}/transformer/config.json")
    config_2b_transformer = load_config(f"{model_path_2b}/transformer/config.json")
    
    transformer_compatible, transformer_diffs = compare_configs(
        config_5b_transformer, config_2b_transformer, "Transformer"
    )
    
    # 2. 比较VAE配置
    print("\n" + "="*60)
    print("2. 比较VAE配置")
    print("="*60)
    
    config_5b_vae = load_config(f"{model_path_5b}/vae/config.json")
    config_2b_vae = load_config(f"{model_path_2b}/vae/config.json")
    
    vae_compatible, vae_diffs = compare_configs(
        config_5b_vae, config_2b_vae, "VAE"
    )
    
    # 3. 测试实际模型
    print("\n" + "="*60)
    print("3. 测试实际模型输入输出形状")
    print("="*60)
    
    shape_test_result = test_transformer_shapes()
    
    # 4. 总结
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)
    
    print(f"\nTransformer配置兼容性: {'✅ 兼容' if transformer_compatible else '❌ 不兼容'}")
    if transformer_diffs:
        print(f"  关键差异: {transformer_diffs}")
    
    print(f"\nVAE配置兼容性: {'✅ 兼容' if vae_compatible else '❌ 不兼容'}")
    if vae_diffs:
        print(f"  关键差异: {vae_diffs}")
    
    if shape_test_result is True:
        print(f"\n输入输出形状: ✅ 兼容")
    elif shape_test_result is False:
        print(f"\n输入输出形状: ❌ 不兼容")
    else:
        print(f"\n输入输出形状: ⚠️  未测试（需要安装diffusers）")
    
    # 5. 兼容性结论
    print("\n" + "="*60)
    print("兼容性结论")
    print("="*60)
    
    # 检查关键参数
    key_compatible = (
        config_5b_transformer['in_channels'] == config_2b_transformer['in_channels'] and
        config_5b_transformer['out_channels'] == config_2b_transformer['out_channels'] and
        config_5b_transformer['sample_frames'] == config_2b_transformer['sample_frames'] and
        config_5b_transformer['sample_height'] == config_2b_transformer['sample_height'] and
        config_5b_transformer['sample_width'] == config_2b_transformer['sample_width'] and
        config_5b_vae['latent_channels'] == config_2b_vae['latent_channels']
    )
    
    if key_compatible:
        print("\n✅✅✅ 关键参数兼容！")
        print("   两个模型的latent空间维度完全一致")
        print("   理论上可以无缝切换transformer模型")
        
        # 检查use_rotary_positional_embeddings
        if config_5b_transformer['use_rotary_positional_embeddings'] != config_2b_transformer['use_rotary_positional_embeddings']:
            print("\n⚠️  注意: use_rotary_positional_embeddings配置不同")
            print(f"   5B: {config_5b_transformer['use_rotary_positional_embeddings']}")
            print(f"   2B: {config_2b_transformer['use_rotary_positional_embeddings']}")
            print("   这可能需要在切换时处理，但不会影响latent兼容性")
    else:
        print("\n❌❌❌ 关键参数不兼容！")
        print("   不能直接切换模型")
    
    # 6. 实现建议
    print("\n" + "="*60)
    print("实现建议")
    print("="*60)
    
    if key_compatible:
        print("\n✅ 可以实现的方案:")
        print("   1. 在denoising loop中动态切换transformer")
        print("   2. 保持latent形状不变")
        print("   3. 处理use_rotary_positional_embeddings的差异（如果需要）")
        print("   4. 共享VAE和Text Encoder")
    else:
        print("\n❌ 需要解决的问题:")
        print("   1. 解决latent维度不兼容的问题")
        print("   2. 或者使用latent转换层")

if __name__ == "__main__":
    main()

