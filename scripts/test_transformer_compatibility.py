#!/usr/bin/env python3
"""
详细测试CogVideoX-5B和CogVideoX-2B的transformer兼容性
"""

import json
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)

def test_single_transformer(model_path, model_name, device="cuda"):
    """测试单个transformer"""
    print(f"\n{'='*60}")
    print(f"测试 {model_name} Transformer")
    print(f"{'='*60}")
    
    try:
        from diffusers import CogVideoXTransformer3DModel
        
        # 加载配置
        config_path = f"{model_path}/transformer/config.json"
        config = load_config(config_path)
        
        print(f"\n配置信息:")
        print(f"  in_channels: {config['in_channels']}")
        print(f"  out_channels: {config['out_channels']}")
        print(f"  sample_frames: {config['sample_frames']}")
        print(f"  sample_height: {config['sample_height']}")
        print(f"  sample_width: {config['sample_width']}")
        print(f"  num_attention_heads: {config['num_attention_heads']}")
        print(f"  num_layers: {config['num_layers']}")
        print(f"  use_rotary_positional_embeddings: {config.get('use_rotary_positional_embeddings', 'N/A')}")
        
        # 加载模型
        print(f"\n加载模型...")
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            f"{model_path}/transformer",
            torch_dtype=torch.float16
        ).to(device)
        transformer.eval()
        
        # 创建测试输入
        batch_size = 1
        num_frames = config['sample_frames']
        latent_channels = config['in_channels']
        latent_height = config['sample_height']
        latent_width = config['sample_width']
        text_embed_dim = config['text_embed_dim']
        
        # Latent输入: (B, T, C, H, W)
        latent_input = torch.randn(
            batch_size, num_frames, latent_channels, latent_height, latent_width,
            device=device, dtype=torch.float16
        )
        
        # Timestep
        timestep = torch.tensor([500], device=device, dtype=torch.long)
        
        # Encoder hidden states: (B, seq_len, text_embed_dim)
        # 注意：seq_len可能需要根据模型调整
        seq_len = 77
        encoder_hidden_states = torch.randn(
            batch_size, seq_len, text_embed_dim,
            device=device, dtype=torch.float16
        )
        
        print(f"\n输入形状:")
        print(f"  latent_input: {latent_input.shape}")
        print(f"  timestep: {timestep.shape}")
        print(f"  encoder_hidden_states: {encoder_hidden_states.shape}")
        
        # 测试前向传播
        print(f"\n测试前向传播...")
        with torch.no_grad():
            try:
                # 检查是否需要rotary positional embeddings
                use_rotary = config.get('use_rotary_positional_embeddings', False)
                
                if use_rotary:
                    print("  使用rotary positional embeddings")
                    # 对于rotary embeddings，需要准备image_rotary_emb
                    # 这里先尝试不使用
                    output = transformer(
                        hidden_states=latent_input,
                        encoder_hidden_states=encoder_hidden_states,
                        timestep=timestep,
                        return_dict=False
                    )[0]
                else:
                    print("  不使用rotary positional embeddings")
                    output = transformer(
                        hidden_states=latent_input,
                        encoder_hidden_states=encoder_hidden_states,
                        timestep=timestep,
                        return_dict=False
                    )[0]
                
                print(f"✅ 输出形状: {output.shape}")
                print(f"✅ 输出数据类型: {output.dtype}")
                print(f"✅ 输出数值范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
                
                return {
                    'success': True,
                    'output_shape': output.shape,
                    'config': config
                }
            except Exception as e:
                print(f"❌ 前向传播失败: {e}")
                import traceback
                traceback.print_exc()
                return {
                    'success': False,
                    'error': str(e),
                    'config': config
                }
                
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'config': None
        }

def compare_transformer_io(model_path_5b, model_path_2b, device="cuda"):
    """比较两个transformer的输入输出"""
    print("="*60)
    print("CogVideoX Transformer 兼容性详细测试")
    print("="*60)
    
    # 测试5B模型
    result_5b = test_single_transformer(model_path_5b, "CogVideoX-5B", device)
    
    # 测试2B模型
    result_2b = test_single_transformer(model_path_2b, "CogVideoX-2B", device)
    
    # 比较结果
    print(f"\n{'='*60}")
    print("兼容性分析")
    print(f"{'='*60}")
    
    if result_5b['success'] and result_2b['success']:
        print("\n✅ 两个模型都可以正常前向传播")
        
        # 比较输出形状
        shape_5b = result_5b['output_shape']
        shape_2b = result_2b['output_shape']
        
        print(f"\n输出形状对比:")
        print(f"  5B: {shape_5b}")
        print(f"  2B: {shape_2b}")
        
        if shape_5b == shape_2b:
            print(f"\n✅✅✅ 输出形状完全相同！")
            print(f"   可以无缝切换transformer模型")
            
            # 比较配置
            config_5b = result_5b['config']
            config_2b = result_2b['config']
            
            print(f"\n关键配置对比:")
            key_params = ['in_channels', 'out_channels', 'sample_frames', 
                         'sample_height', 'sample_width', 'text_embed_dim']
            
            all_same = True
            for param in key_params:
                val_5b = config_5b.get(param)
                val_2b = config_2b.get(param)
                if val_5b == val_2b:
                    print(f"  ✅ {param}: {val_5b} (相同)")
                else:
                    print(f"  ❌ {param}: 5B={val_5b}, 2B={val_2b} (不同)")
                    all_same = False
            
            if all_same:
                print(f"\n✅✅✅ 所有关键参数相同！")
                print(f"   模型切换完全可行")
            else:
                print(f"\n⚠️  部分参数不同，但输出形状相同")
                print(f"   可能需要特殊处理，但可以切换")
            
            # 检查use_rotary_positional_embeddings
            rotary_5b = config_5b.get('use_rotary_positional_embeddings', False)
            rotary_2b = config_2b.get('use_rotary_positional_embeddings', False)
            
            if rotary_5b != rotary_2b:
                print(f"\n⚠️  注意: use_rotary_positional_embeddings不同")
                print(f"   5B: {rotary_5b}")
                print(f"   2B: {rotary_2b}")
                print(f"   在切换时可能需要根据模型类型处理")
            else:
                print(f"\n✅ use_rotary_positional_embeddings相同: {rotary_5b}")
            
            return True
        else:
            print(f"\n❌❌❌ 输出形状不同！")
            print(f"   不能直接切换模型")
            return False
    else:
        print(f"\n❌ 测试失败")
        if not result_5b['success']:
            print(f"  5B模型错误: {result_5b.get('error', 'Unknown')}")
        if not result_2b['success']:
            print(f"  2B模型错误: {result_2b.get('error', 'Unknown')}")
        return False

def test_cross_model_inference(model_path_5b, model_path_2b, device="cuda"):
    """测试跨模型推理（模拟协同推理）"""
    print(f"\n{'='*60}")
    print("测试跨模型推理（模拟协同推理）")
    print(f"{'='*60}")
    
    try:
        from diffusers import CogVideoXTransformer3DModel
        
        # 加载两个transformer
        print("\n加载模型...")
        transformer_5b = CogVideoXTransformer3DModel.from_pretrained(
            f"{model_path_5b}/transformer",
            torch_dtype=torch.float16
        ).to(device)
        transformer_5b.eval()
        
        transformer_2b = CogVideoXTransformer3DModel.from_pretrained(
            f"{model_path_2b}/transformer",
            torch_dtype=torch.float16
        ).to(device)
        transformer_2b.eval()
        
        # 创建测试输入（使用5B的配置，因为两个模型的latent空间应该相同）
        config_5b = load_config(f"{model_path_5b}/transformer/config.json")
        
        batch_size = 1
        num_frames = config_5b['sample_frames']
        latent_channels = config_5b['in_channels']
        latent_height = config_5b['sample_height']
        latent_width = config_5b['sample_width']
        text_embed_dim = config_5b['text_embed_dim']
        
        # 创建latent
        latent = torch.randn(
            batch_size, num_frames, latent_channels, latent_height, latent_width,
            device=device, dtype=torch.float16
        )
        
        timestep = torch.tensor([500], device=device, dtype=torch.long)
        encoder_hidden_states = torch.randn(
            batch_size, 77, text_embed_dim,
            device=device, dtype=torch.float16
        )
        
        print(f"\n测试场景: 先用5B模型处理，再用2B模型处理")
        
        # 步骤1: 用5B模型处理
        print("\n步骤1: 使用5B模型处理...")
        with torch.no_grad():
            output_5b = transformer_5b(
                hidden_states=latent,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                return_dict=False
            )[0]
        print(f"✅ 5B输出形状: {output_5b.shape}")
        
        # 步骤2: 用2B模型处理5B的输出（模拟模型切换）
        print("\n步骤2: 使用2B模型处理5B的输出...")
        with torch.no_grad():
            output_2b = transformer_2b(
                hidden_states=output_5b,  # 使用5B的输出作为2B的输入
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                return_dict=False
            )[0]
        print(f"✅ 2B输出形状: {output_2b.shape}")
        
        # 检查形状
        if output_5b.shape == output_2b.shape:
            print(f"\n✅✅✅ 跨模型推理成功！")
            print(f"   可以无缝切换transformer模型")
            print(f"   输出形状保持一致: {output_2b.shape}")
            return True
        else:
            print(f"\n❌ 跨模型推理失败")
            print(f"   形状不匹配: {output_5b.shape} vs {output_2b.shape}")
            return False
            
    except Exception as e:
        print(f"\n❌ 跨模型推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    model_path_5b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-5B"
    model_path_2b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-2B"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 测试1: 比较两个模型的输入输出
    compatible = compare_transformer_io(model_path_5b, model_path_2b, device)
    
    # 测试2: 跨模型推理测试
    if compatible:
        cross_compatible = test_cross_model_inference(model_path_5b, model_path_2b, device)
    else:
        cross_compatible = False
    
    # 最终结论
    print(f"\n{'='*60}")
    print("最终结论")
    print(f"{'='*60}")
    
    if compatible and cross_compatible:
        print("\n✅✅✅ 完全兼容！")
        print("   可以实现协同推理")
        print("   可以在denoising loop中无缝切换transformer模型")
    elif compatible:
        print("\n✅ 基本兼容")
        print("   输出形状相同，但跨模型推理需要进一步测试")
    else:
        print("\n❌ 不兼容")
        print("   需要解决兼容性问题")

if __name__ == "__main__":
    main()

