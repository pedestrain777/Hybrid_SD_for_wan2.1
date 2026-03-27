#!/usr/bin/env python3
"""
测试HybridVideoInferencePipeline的正确性
"""

import sys
import os
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.hybrid_sd.inference_pipeline import HybridVideoInferencePipeline

class MockArgs:
    """模拟args对象"""
    def __init__(self, steps, enable_xformers=False, use_dpm_solver=False):
        self.steps = steps
        self.enable_xformers_memory_efficient_attention = enable_xformers
        self.use_dpm_solver = use_dpm_solver
        self.logger = None

def test_pipeline_initialization():
    """测试Pipeline初始化"""
    print("="*60)
    print("测试1: HybridVideoInferencePipeline初始化")
    print("="*60)
    
    try:
        model_path_5b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-5B"
        model_path_2b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-2B"
        
        weight_folders = [model_path_5b, model_path_2b]
        seed = 42
        device = "cuda" if torch.cuda.is_available() else "cpu"
        args = MockArgs(steps=[10, 15])  # 5B用10步，2B用15步
        
        print(f"\n创建HybridVideoInferencePipeline...")
        print(f"  模型路径: {weight_folders}")
        print(f"  步数配置: {args.steps}")
        print(f"  设备: {device}")
        
        pipeline = HybridVideoInferencePipeline(
            weight_folders=weight_folders,
            seed=seed,
            device=device,
            args=args
        )
        
        print("✅ Pipeline创建成功")
        
        # 测试set_pipe_and_generator
        print("\n测试set_pipe_and_generator...")
        pipeline.set_pipe_and_generator()
        
        print("✅ set_pipe_and_generator成功")
        print(f"  Pipeline类型: {type(pipeline.pipe)}")
        print(f"  Transformer数量: {len(pipeline.pipe.transformers) if pipeline.pipe.transformers else 0}")
        print(f"  总步数: {pipeline.total_step}")
        
        return True, pipeline
        
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_step_config():
    """测试步数配置"""
    print("\n" + "="*60)
    print("测试2: 步数配置")
    print("="*60)
    
    try:
        model_path_5b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-5B"
        model_path_2b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-2B"
        
        weight_folders = [model_path_5b, model_path_2b]
        args = MockArgs(steps=[10, 15])
        
        pipeline = HybridVideoInferencePipeline(
            weight_folders=weight_folders,
            seed=42,
            device="cuda" if torch.cuda.is_available() else "cpu",
            args=args
        )
        
        total_step, step_config = pipeline.get_step_config(args)
        
        print(f"\n步数配置结果:")
        print(f"  总步数: {total_step}")
        print(f"  步数分配: {args.steps}")
        print(f"  前10步: 使用模型 {step_config['step'][0]} ({step_config['name'][0]})")
        print(f"  后15步: 使用模型 {step_config['step'][10]} ({step_config['name'][1]})")
        
        # 验证配置
        assert total_step == 25, f"总步数应该是25，实际是{total_step}"
        assert step_config["step"][0] == 0, "第0步应该使用模型0"
        assert step_config["step"][10] == 1, "第10步应该使用模型1"
        assert step_config["name"][0] == "THUDM--CogVideoX-5B", "模型0名称不正确"
        assert step_config["name"][1] == "THUDM--CogVideoX-2B", "模型1名称不正确"
        
        print("✅ 步数配置正确")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """测试模型加载"""
    print("\n" + "="*60)
    print("测试3: 模型加载")
    print("="*60)
    
    try:
        model_path_5b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-5B"
        model_path_2b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-2B"
        
        weight_folders = [model_path_5b, model_path_2b]
        args = MockArgs(steps=[10, 15])
        
        pipeline = HybridVideoInferencePipeline(
            weight_folders=weight_folders,
            seed=42,
            device="cuda" if torch.cuda.is_available() else "cpu",
            args=args
        )
        
        print("\n加载模型...")
        pipeline.set_pipe_and_generator()
        
        # 检查模型
        print("\n检查模型:")
        print(f"  Text Encoder类型: {type(pipeline.pipe.text_encoder)}")
        print(f"  VAE类型: {type(pipeline.pipe.vae)}")
        print(f"  Transformer数量: {len(pipeline.pipe.transformers)}")
        for i, transformer in enumerate(pipeline.pipe.transformers):
            print(f"    Transformer {i}: {type(transformer)}")
            print(f"      use_rotary: {transformer.config.use_rotary_positional_embeddings}")
        
        # 检查step_config
        step_config = pipeline.pipe.step_config
        print(f"\n  Step配置:")
        print(f"    总步数: {pipeline.total_step}")
        print(f"    模型映射: {step_config['name']}")
        
        # 验证
        assert pipeline.pipe.transformers is not None, "transformers应该被设置"
        assert len(pipeline.pipe.transformers) == 2, "应该有2个transformer"
        assert pipeline.pipe.step_config is not None, "step_config应该被设置"
        
        print("✅ 模型加载正确")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pipeline_structure():
    """测试Pipeline结构"""
    print("\n" + "="*60)
    print("测试4: Pipeline结构")
    print("="*60)
    
    try:
        model_path_5b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-5B"
        model_path_2b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-2B"
        
        weight_folders = [model_path_5b, model_path_2b]
        args = MockArgs(steps=[10, 15])
        
        pipeline = HybridVideoInferencePipeline(
            weight_folders=weight_folders,
            seed=42,
            device="cuda" if torch.cuda.is_available() else "cpu",
            args=args
        )
        
        pipeline.set_pipe_and_generator()
        
        # 检查关键属性
        print("\n检查Pipeline结构:")
        print(f"  pipe: {type(pipeline.pipe)}")
        print(f"  generator: {type(pipeline.generator)}")
        print(f"  total_step: {pipeline.total_step}")
        print(f"  device: {pipeline.device}")
        print(f"  weight_folders: {pipeline.weight_folders}")
        
        # 检查pipe的关键属性
        print(f"\n检查pipe属性:")
        print(f"  transformers: {pipeline.pipe.transformers is not None}")
        print(f"  step_config: {pipeline.pipe.step_config is not None}")
        print(f"  text_encoder: {pipeline.pipe.text_encoder is not None}")
        print(f"  vae: {pipeline.pipe.vae is not None}")
        print(f"  scheduler: {pipeline.pipe.scheduler is not None}")
        
        # 验证
        assert pipeline.pipe is not None, "pipe应该被创建"
        assert pipeline.generator is not None, "generator应该被创建"
        assert pipeline.total_step > 0, "total_step应该大于0"
        
        print("✅ Pipeline结构正确")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("="*60)
    print("HybridVideoInferencePipeline 正确性验证")
    print("="*60)
    
    results = []
    
    # 测试1: Pipeline初始化
    success, pipeline = test_pipeline_initialization()
    results.append(("Pipeline初始化", success))
    
    # 测试2: 步数配置
    success = test_step_config()
    results.append(("步数配置", success))
    
    # 测试3: 模型加载
    success = test_model_loading()
    results.append(("模型加载", success))
    
    # 测试4: Pipeline结构
    success = test_pipeline_structure()
    results.append(("Pipeline结构", success))
    
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
        print("HybridVideoInferencePipeline实现正确")
    else:
        print("\n❌ 部分测试失败，请检查实现")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

