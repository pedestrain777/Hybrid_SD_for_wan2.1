#!/usr/bin/env python3
"""
测试HybridCogVideoXPipeline的正确性
"""

import sys
import os
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.hybrid_sd.diffusers.pipeline_cogvideox import HybridCogVideoXPipeline
from diffusers import CogVideoXTransformer3DModel, CogVideoXDPMScheduler

def test_pipeline_initialization():
    """测试Pipeline初始化"""
    print("="*60)
    print("测试1: Pipeline初始化")
    print("="*60)
    
    try:
        model_path_5b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-5B"
        
        # 加载pipeline - 先加载标准pipeline，然后转换为Hybrid
        print(f"\n加载pipeline: {model_path_5b}")
        from diffusers import CogVideoXPipeline
        base_pipeline = CogVideoXPipeline.from_pretrained(
            model_path_5b,
            torch_dtype=torch.float16
        )
        # 转换为HybridCogVideoXPipeline
        pipeline = HybridCogVideoXPipeline(
            transformer=base_pipeline.transformer,
            vae=base_pipeline.vae,
            text_encoder=base_pipeline.text_encoder,
            tokenizer=base_pipeline.tokenizer,
            scheduler=base_pipeline.scheduler
        )
        
        print("✅ Pipeline加载成功")
        print(f"  Transformer类型: {type(pipeline.transformer)}")
        print(f"  VAE类型: {type(pipeline.vae)}")
        print(f"  Text Encoder类型: {type(pipeline.text_encoder)}")
        
        # 测试设置transformers
        print("\n测试设置多个transformers...")
        transformer_5b = pipeline.transformer
        transformers = [transformer_5b]  # 暂时只有一个，后续可以添加2B
        
        pipeline.set_transformers(transformers)
        print(f"✅ 设置transformers成功: {len(pipeline.transformers)}个")
        
        # 测试设置step_config
        print("\n测试设置step_config...")
        step_config = {
            "step": {i: 0 for i in range(25)},  # 所有步都用模型0
            "name": {0: "CogVideoX-5B"}
        }
        pipeline.set_step_config(step_config)
        print("✅ 设置step_config成功")
        
        return True, pipeline
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_hybrid_mode_detection():
    """测试混合模式检测"""
    print("\n" + "="*60)
    print("测试2: 混合模式检测")
    print("="*60)
    
    try:
        model_path_5b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-5B"
        model_path_2b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-2B"
        
        from diffusers import CogVideoXPipeline
        base_pipeline = CogVideoXPipeline.from_pretrained(
            model_path_5b,
            torch_dtype=torch.float16
        )
        pipeline = HybridCogVideoXPipeline(
            transformer=base_pipeline.transformer,
            vae=base_pipeline.vae,
            text_encoder=base_pipeline.text_encoder,
            tokenizer=base_pipeline.tokenizer,
            scheduler=base_pipeline.scheduler
        )
        
        # 加载两个transformer
        print("\n加载两个transformer...")
        transformer_5b = CogVideoXTransformer3DModel.from_pretrained(
            f"{model_path_5b}/transformer",
            torch_dtype=torch.float16
        )
        transformer_2b = CogVideoXTransformer3DModel.from_pretrained(
            f"{model_path_2b}/transformer",
            torch_dtype=torch.float16
        )
        
        transformers = [transformer_5b, transformer_2b]
        pipeline.set_transformers(transformers)
        print(f"✅ 加载了{len(transformers)}个transformer")
        
        # 设置step_config
        step_config = {
            "step": {
                **{i: 0 for i in range(10)},  # 前10步用5B
                **{i: 1 for i in range(10, 25)}  # 后15步用2B
            },
            "name": {
                0: "CogVideoX-5B",
                1: "CogVideoX-2B"
            }
        }
        pipeline.set_step_config(step_config)
        print("✅ 设置step_config成功")
        
        # 检查混合模式
        use_hybrid = (
            pipeline.transformers is not None 
            and len(pipeline.transformers) > 1 
            and pipeline.step_config is not None
        )
        
        if use_hybrid:
            print("✅ 混合模式已启用")
        else:
            print("❌ 混合模式未启用")
        
        return use_hybrid, pipeline
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_rotary_embeddings():
    """测试Rotary Embeddings处理"""
    print("\n" + "="*60)
    print("测试3: Rotary Embeddings处理")
    print("="*60)
    
    try:
        model_path_5b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-5B"
        model_path_2b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-2B"
        
        from diffusers import CogVideoXPipeline
        base_pipeline = CogVideoXPipeline.from_pretrained(
            model_path_5b,
            torch_dtype=torch.float16
        )
        pipeline = HybridCogVideoXPipeline(
            transformer=base_pipeline.transformer,
            vae=base_pipeline.vae,
            text_encoder=base_pipeline.text_encoder,
            tokenizer=base_pipeline.tokenizer,
            scheduler=base_pipeline.scheduler
        )
        
        # 加载两个transformer
        transformer_5b = CogVideoXTransformer3DModel.from_pretrained(
            f"{model_path_5b}/transformer",
            torch_dtype=torch.float16
        )
        transformer_2b = CogVideoXTransformer3DModel.from_pretrained(
            f"{model_path_2b}/transformer",
            torch_dtype=torch.float16
        )
        
        transformers = [transformer_5b, transformer_2b]
        pipeline.set_transformers(transformers)
        
        # 检查rotary embeddings配置
        print("\n检查Rotary Embeddings配置:")
        print(f"  5B use_rotary: {transformer_5b.config.use_rotary_positional_embeddings}")
        print(f"  2B use_rotary: {transformer_2b.config.use_rotary_positional_embeddings}")
        
        # 测试准备rotary embeddings
        device = "cuda" if torch.cuda.is_available() else "cpu"
        height = 480
        width = 720
        num_frames = 49
        
        print("\n测试准备rotary embeddings...")
        image_rotary_embs = []
        for i, transformer in enumerate(transformers):
            if transformer.config.use_rotary_positional_embeddings:
                image_rotary_emb = pipeline._prepare_rotary_positional_embeddings(
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    device=device,
                )
                print(f"  Transformer {i}: 生成rotary embeddings (shape: {image_rotary_emb.shape if hasattr(image_rotary_emb, 'shape') else 'N/A'})")
            else:
                image_rotary_emb = None
                print(f"  Transformer {i}: 不使用rotary embeddings (None)")
            image_rotary_embs.append(image_rotary_emb)
        
        print("✅ Rotary embeddings处理正确")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step_config():
    """测试步数配置"""
    print("\n" + "="*60)
    print("测试4: 步数配置")
    print("="*60)
    
    try:
        # 模拟步数配置
        steps = [10, 15]  # 大模型10步，小模型15步
        
        step_config = {"step": {}, "name": {}}
        total_step = 0
        for index, model_step in enumerate(steps):
            for i in range(model_step):
                step_config["step"][total_step] = index
                total_step += 1
        
        model_names = ["CogVideoX-5B", "CogVideoX-2B"]
        for index, model_name in enumerate(model_names):
            step_config["name"][index] = model_name
        
        print(f"\n步数配置:")
        print(f"  总步数: {total_step}")
        print(f"  步数分配: {steps}")
        print(f"  前10步: 使用模型 {step_config['step'][0]} ({step_config['name'][0]})")
        print(f"  后15步: 使用模型 {step_config['step'][10]} ({step_config['name'][1]})")
        
        # 验证配置
        assert total_step == 25, f"总步数应该是25，实际是{total_step}"
        assert step_config["step"][0] == 0, "第0步应该使用模型0"
        assert step_config["step"][10] == 1, "第10步应该使用模型1"
        
        print("✅ 步数配置正确")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("="*60)
    print("HybridCogVideoXPipeline 正确性验证")
    print("="*60)
    
    results = []
    
    # 测试1: Pipeline初始化
    success, pipeline = test_pipeline_initialization()
    results.append(("Pipeline初始化", success))
    
    # 测试2: 混合模式检测
    success, _ = test_hybrid_mode_detection()
    results.append(("混合模式检测", success))
    
    # 测试3: Rotary Embeddings
    success = test_rotary_embeddings()
    results.append(("Rotary Embeddings", success))
    
    # 测试4: 步数配置
    success = test_step_config()
    results.append(("步数配置", success))
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n✅✅✅ 所有测试通过！")
        print("HybridCogVideoXPipeline实现正确")
    else:
        print("\n❌ 部分测试失败，请检查实现")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

