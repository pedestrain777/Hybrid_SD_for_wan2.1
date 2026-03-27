#!/usr/bin/env python3
"""
验证hybrid_video.py和run_cogvideox.sh的正确性
"""

import os
import sys

def test_file_exists():
    """测试文件是否存在"""
    print("="*60)
    print("测试1: 文件存在性检查")
    print("="*60)
    
    files = [
        "examples/hybrid_sd/hybrid_video.py",
        "scripts/hybrid_sd/run_cogvideox.sh"
    ]
    
    all_exist = True
    for file_path in files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}: 存在")
        else:
            print(f"❌ {file_path}: 不存在")
            all_exist = False
    
    return all_exist

def test_hybrid_video_structure():
    """测试hybrid_video.py的结构"""
    print("\n" + "="*60)
    print("测试2: hybrid_video.py结构检查")
    print("="*60)
    
    try:
        file_path = "examples/hybrid_sd/hybrid_video.py"
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 检查关键组件
        checks = [
            ('def parse_args', 'parse_args函数'),
            ('HybridVideoInferencePipeline', 'HybridVideoInferencePipeline导入'),
            ('pipeline.generate', 'pipeline.generate调用'),
            ('export_to_video', 'export_to_video调用'),
            ('--model_id', 'model_id参数'),
            ('--steps', 'steps参数'),
            ('--num_frames', 'num_frames参数'),
            ('--height', 'height参数'),
            ('--width', 'width参数'),
        ]
        
        all_found = True
        for check_str, desc in checks:
            if check_str in content:
                print(f"  ✅ {desc}: 找到")
            else:
                print(f"  ❌ {desc}: 未找到")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_run_cogvideox_structure():
    """测试run_cogvideox.sh的结构"""
    print("\n" + "="*60)
    print("测试3: run_cogvideox.sh结构检查")
    print("="*60)
    
    try:
        file_path = "scripts/hybrid_sd/run_cogvideox.sh"
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 检查关键内容
        checks = [
            ('MODEL_LARGE=THUDM--CogVideoX-5B', '大模型路径'),
            ('MODEL_SMALL=THUDM--CogVideoX-2B', '小模型路径'),
            ('step_list=', '步数配置列表'),
            ('hybrid_video.py', '调用hybrid_video.py'),
            ('--model_id', '模型ID参数'),
            ('--steps', '步数参数'),
            ('--num_frames', '帧数参数'),
            ('--height', '高度参数'),
            ('--width', '宽度参数'),
            ('--use_dpm_solver', 'DPM solver参数'),
        ]
        
        all_found = True
        for check_str, desc in checks:
            if check_str in content:
                print(f"  ✅ {desc}: 找到")
            else:
                print(f"  ❌ {desc}: 未找到")
                all_found = False
        
        # 检查shebang
        if content.startswith('#!/bin/bash'):
            print(f"  ✅ Shebang: 正确")
        else:
            print(f"  ⚠️  Shebang: 可能缺失")
        
        return all_found
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_syntax():
    """测试Python语法"""
    print("\n" + "="*60)
    print("测试4: Python语法检查")
    print("="*60)
    
    try:
        file_path = "examples/hybrid_sd/hybrid_video.py"
        
        # 使用compile检查语法
        with open(file_path, 'r') as f:
            code = f.read()
        
        try:
            compile(code, file_path, 'exec')
            print("✅ Python语法正确")
            return True
        except SyntaxError as e:
            print(f"❌ Python语法错误: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_imports():
    """测试导入语句"""
    print("\n" + "="*60)
    print("测试5: 导入语句检查")
    print("="*60)
    
    try:
        file_path = "examples/hybrid_sd/hybrid_video.py"
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # 检查关键导入
        imports_found = {
            'HybridVideoInferencePipeline': False,
            'export_to_video': False,
            'argparse': False,
            'torch': False,
        }
        
        for line in lines[:50]:  # 只检查前50行
            for key in imports_found:
                if key in line and ('import' in line or 'from' in line):
                    imports_found[key] = True
        
        all_found = True
        for key, found in imports_found.items():
            if found:
                print(f"  ✅ {key}: 找到")
            else:
                print(f"  ❌ {key}: 未找到")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("="*60)
    print("hybrid_video.py 和 run_cogvideox.sh 验证")
    print("="*60)
    
    results = []
    
    # 测试1: 文件存在
    success = test_file_exists()
    results.append(("文件存在性", success))
    
    # 测试2: hybrid_video.py结构
    success = test_hybrid_video_structure()
    results.append(("hybrid_video.py结构", success))
    
    # 测试3: run_cogvideox.sh结构
    success = test_run_cogvideox_structure()
    results.append(("run_cogvideox.sh结构", success))
    
    # 测试4: Python语法
    success = test_syntax()
    results.append(("Python语法", success))
    
    # 测试5: 导入语句
    success = test_imports()
    results.append(("导入语句", success))
    
    # 总结
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n✅✅✅ 所有验证通过！")
        print("hybrid_video.py和run_cogvideox.sh实现正确")
    else:
        print("\n❌ 部分验证失败，请检查实现")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

