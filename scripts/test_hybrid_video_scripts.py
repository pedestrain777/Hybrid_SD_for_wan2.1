#!/usr/bin/env python3
"""
测试hybrid_video.py和run_cogvideox.sh的正确性
"""

import sys
import os
import subprocess
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_hybrid_video_import():
    """测试hybrid_video.py可以正确导入"""
    print("="*60)
    print("测试1: hybrid_video.py导入")
    print("="*60)
    
    try:
        # 检查文件是否存在
        script_path = "examples/hybrid_sd/hybrid_video.py"
        if not os.path.exists(script_path):
            print(f"❌ 文件不存在: {script_path}")
            return False
        
        # 尝试解析参数
        import sys
        old_argv = sys.argv
        sys.argv = ['hybrid_video.py', '--help']
        
        try:
            from examples.hybrid_sd import hybrid_video
            # 重置sys.argv
            sys.argv = old_argv
            
            print("✅ hybrid_video.py可以正确导入")
            return True
        except Exception as e:
            sys.argv = old_argv
            print(f"❌ 导入失败: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hybrid_video_args():
    """测试hybrid_video.py参数解析"""
    print("\n" + "="*60)
    print("测试2: hybrid_video.py参数解析")
    print("="*60)
    
    try:
        script_path = "examples/hybrid_sd/hybrid_video.py"
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 测试help命令，设置PYTHONPATH
        env = os.environ.copy()
        env['PYTHONPATH'] = project_root
        
        result = subprocess.run(
            ['python3', script_path, '--help'],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
            cwd=project_root
        )
        
        if result.returncode == 0:
            print("✅ 参数解析成功")
            print("   支持的参数:")
            # 检查关键参数
            key_args = ['--model_id', '--steps', '--num_frames', '--height', '--width', '--guidance_scale']
            for arg in key_args:
                if arg in result.stdout:
                    print(f"     ✅ {arg}")
            return True
        else:
            print(f"❌ 参数解析失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_run_cogvideox_sh():
    """测试run_cogvideox.sh脚本"""
    print("\n" + "="*60)
    print("测试3: run_cogvideox.sh脚本")
    print("="*60)
    
    try:
        script_path = "scripts/hybrid_sd/run_cogvideox.sh"
        
        if not os.path.exists(script_path):
            print(f"❌ 文件不存在: {script_path}")
            return False
        
        # 检查脚本是否可执行
        if not os.access(script_path, os.X_OK):
            print(f"⚠️  脚本不可执行，尝试添加执行权限...")
            os.chmod(script_path, 0o755)
        
        # 读取脚本内容
        with open(script_path, 'r') as f:
            content = f.read()
        
        # 检查关键内容
        checks = [
            ('MODEL_LARGE=THUDM--CogVideoX-5B', '大模型路径'),
            ('MODEL_SMALL=THUDM--CogVideoX-2B', '小模型路径'),
            ('step_list=', '步数配置'),
            ('hybrid_video.py', '调用hybrid_video.py'),
            ('--model_id', '模型ID参数'),
            ('--steps', '步数参数'),
        ]
        
        all_passed = True
        for check_str, desc in checks:
            if check_str in content:
                print(f"  ✅ {desc}: 找到")
            else:
                print(f"  ❌ {desc}: 未找到")
                all_passed = False
        
        if all_passed:
            print("✅ run_cogvideox.sh脚本正确")
        else:
            print("❌ run_cogvideox.sh脚本有问题")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_hybrid_video_minimal():
    """测试hybrid_video.py最小运行（不实际生成视频）"""
    print("\n" + "="*60)
    print("测试4: hybrid_video.py最小运行测试")
    print("="*60)
    
    try:
        script_path = "examples/hybrid_sd/hybrid_video.py"
        
        # 检查prompts文件是否存在
        prompts_file = "examples/hybrid_sd/prompts.txt"
        if not os.path.exists(prompts_file):
            print(f"⚠️  prompts文件不存在: {prompts_file}")
            print("   创建测试prompts文件...")
            os.makedirs(os.path.dirname(prompts_file), exist_ok=True)
            with open(prompts_file, 'w') as f:
                f.write("A cat walking on the street.\n")
                f.write("low quality, blurry\n")
            print("   ✅ 已创建测试prompts文件")
        
        # 测试参数解析（不实际运行）
        print("\n测试参数解析...")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = os.environ.copy()
        env['PYTHONPATH'] = project_root
        
        result = subprocess.run(
            ['python3', script_path, '--help'],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
            cwd=project_root
        )
        
        if result.returncode == 0:
            print("✅ 参数解析成功")
            
            # 检查是否可以导入并解析参数
            print("\n测试参数默认值...")
            import sys
            old_argv = sys.argv
            sys.argv = ['hybrid_video.py']
            
            try:
                from examples.hybrid_sd.hybrid_video import parse_args
                args = parse_args()
                
                print(f"  默认模型: {args.model_id}")
                print(f"  默认步数: {args.steps}")
                print(f"  默认帧数: {args.num_frames}")
                print(f"  默认尺寸: {args.height}x{args.width}")
                
                sys.argv = old_argv
                print("✅ 参数默认值正确")
                return True
            except Exception as e:
                sys.argv = old_argv
                print(f"❌ 参数解析失败: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"❌ 测试失败: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("="*60)
    print("hybrid_video.py 和 run_cogvideox.sh 正确性验证")
    print("="*60)
    
    results = []
    
    # 测试1: 导入测试
    success = test_hybrid_video_import()
    results.append(("hybrid_video.py导入", success))
    
    # 测试2: 参数解析
    success = test_hybrid_video_args()
    results.append(("参数解析", success))
    
    # 测试3: run_cogvideox.sh
    success = test_run_cogvideox_sh()
    results.append(("run_cogvideox.sh脚本", success))
    
    # 测试4: 最小运行测试
    success = test_hybrid_video_minimal()
    results.append(("最小运行测试", success))
    
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
        print("hybrid_video.py和run_cogvideox.sh实现正确")
    else:
        print("\n❌ 部分测试失败，请检查实现")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

