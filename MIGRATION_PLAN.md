# Hybrid-sd_wan 迁移计划：从 CogVideoX 到 Wan 系列

## 📋 项目概述

**当前状态：** Hybrid-sd_wan 使用 CogVideoX-5B 和 CogVideoX-2B 进行协同推理
**目标状态：** 迁移到 Wan 系列模型（Wan2.1-T2V-14B 和 Wan2.1-T2V-1.3B）

---

## 🔍 当前架构分析

### 核心代码文件结构

```
Hybrid-sd_wan/
├── compression/
│   └── hybrid_sd/
│       ├── inference_pipeline.py          # 主推理Pipeline（核心）
│       └── diffusers/
│           └── pipeline_cogvideox.py      # CogVideoX Hybrid Pipeline（需修改）
├── scripts/
│   ├── test_hybrid_video_inference_pipeline.py  # 测试脚本
│   └── test_hybrid_cogvideox_pipeline.py        # CogVideoX测试脚本
└── configs/                               # 配置文件
```

### 关键类和方法

#### 1. **HybridVideoInferencePipeline** (`inference_pipeline.py:648-927`)
   - **作用：** 主推理Pipeline，负责加载模型和生成视频
   - **关键方法：**
     - `__init__()`: 初始化，设置模型路径、设备等
     - `set_pipe_and_generator()`: 加载模型组件（Text Encoder, VAE, Transformers）
     - `get_step_config()`: 配置步数分配（哪些步用大模型，哪些步用小模型）
     - `generate()`: 生成视频的主方法

#### 2. **HybridWanPipeline** (`pipeline_cogvideox.py:44-`)
   - **作用：** 实现协同推理的核心逻辑
   - **关键方法：**
     - `set_transformers()`: 设置多个transformer
     - `set_step_config()`: 设置步数配置
     - `set_scheduler_configs()`: 设置调度器配置
     - `__call__()`: 执行推理，在不同步数切换模型

---

## 🎯 迁移计划

### 阶段1：模型组件适配（核心修改）

#### 1.1 修改 `inference_pipeline.py` 中的 `HybridVideoInferencePipeline`

**需要修改的部分：**

```python
# 当前代码（CogVideoX）：
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers import CogVideoXPipeline

# 修改为（Wan）：
from transformers import T5EncoderModel, T5Tokenizer  # Wan也使用T5
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers import WanPipeline
```

**具体修改位置：**
- **Line 688-690**: 导入语句
- **Line 694-700**: Text Encoder加载（保持不变，Wan也用T5）
- **Line 704-710**: VAE加载
  ```python
  # 修改前：
  vae = AutoencoderKLCogVideoX.from_pretrained(...)

  # 修改后：
  vae = AutoencoderKLWan.from_pretrained(...)
  ```
- **Line 717-722**: Transformer加载
  ```python
  # 修改前：
  transformer = CogVideoXTransformer3DModel.from_pretrained(...)

  # 修改后：
  transformer = WanTransformer3DModel.from_pretrained(...)
  ```
- **Line 748-755**: Base Pipeline创建
  ```python
  # 修改前：
  base_pipeline = CogVideoXPipeline.from_pretrained(...)

  # 修改后：
  base_pipeline = WanPipeline.from_pretrained(...)
  ```
- **Line 759-765**: Hybrid Pipeline创建
  ```python
  # 修改前：
  self.pipe = HybridCogVideoXPipeline(...)

  # 修改后：
  self.pipe = HybridWanPipeline(...)
  ```
- **Line 774-779**: 额外Transformer加载
  ```python
  # 修改前：
  additional_transformer = CogVideoXTransformer3DModel.from_pretrained(...)

  # 修改后：
  additional_transformer = WanTransformer3DModel.from_pretrained(...)
  ```
- **Line 810-814**: Scheduler配置（如果需要）
  ```python
  # 修改前：
  from diffusers import CogVideoXDPMScheduler

  # 修改后：
  from diffusers import WanDPMScheduler  # 或保持原有scheduler
  ```

#### 1.2 修改 `pipeline_cogvideox.py` 为 `pipeline_wan.py`

**需要修改的部分：**

```python
# Line 31: 导入基类
# 修改前：
from diffusers import WanPipeline  # 这个已经是对的了！

# Line 36-39: 导入Wan模型
# 修改前：
try:
    from diffusers import WanTransformer3DModel
except ImportError:
    WanTransformer3DModel = None

# 这个已经正确了！
```

**注意：** 你的 `pipeline_cogvideox.py` 文件名虽然叫 cogvideox，但实际上已经实现了 `HybridWanPipeline`，继承自 `WanPipeline`。所以这个文件基本不需要大改，只需要：
1. 重命名文件为 `pipeline_wan.py`（可选，为了清晰）
2. 确保所有方法与 WanPipeline 兼容

---

### 阶段2：配置文件和路径修改

#### 2.1 修改模型路径配置

**需要修改的文件：**
- `scripts/test_hybrid_video_inference_pipeline.py`
- 任何使用模型路径的配置文件

```python
# 修改前（CogVideoX）：
model_path_5b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-5B"
model_path_2b = "/data/models/hybridsd_checkpoint/THUDM--CogVideoX-2B"

# 修改后（Wan）：
model_path_14b = "/data/models/hybridsd_checkpoint/Wan-AI--Wan2.1-T2V-14B-Diffusers"
model_path_1_3b = "/data/models/hybridsd_checkpoint/Wan-AI--Wan2.1-T2V-1.3B-Diffusers"
```

#### 2.2 修改步数配置

Wan模型可能需要不同的步数配置：

```python
# CogVideoX 配置示例：
args.steps = [10, 15]  # 5B用10步，2B用15步

# Wan 配置（需要根据实际测试调整）：
args.steps = [20, 30]  # 14B用20步，1.3B用30步
# 或者
args.steps = [15, 20]  # 根据模型性能调整
```

---

### 阶段3：测试和验证

#### 3.1 创建测试脚本

创建新的测试脚本 `scripts/test_hybrid_wan_pipeline.py`：

```python
#!/usr/bin/env python3
"""
测试 Hybrid Wan Pipeline
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.hybrid_sd.inference_pipeline import HybridVideoInferencePipeline

class MockArgs:
    def __init__(self, steps):
        self.steps = steps
        self.enable_xformers_memory_efficient_attention = False
        self.use_dpm_solver = False
        self.vae_device = None

def test_wan_pipeline():
    print("="*60)
    print("测试 Hybrid Wan Pipeline")
    print("="*60)

    # Wan模型路径
    model_path_14b = "/path/to/Wan2.1-T2V-14B-Diffusers"
    model_path_1_3b = "/path/to/Wan2.1-T2V-1.3B-Diffusers"

    weight_folders = [model_path_14b, model_path_1_3b]
    args = MockArgs(steps=[20, 30])

    pipeline = HybridVideoInferencePipeline(
        weight_folders=weight_folders,
        seed=42,
        device="cuda",
        args=args
    )

    pipeline.set_pipe_and_generator()

    # 测试生成
    prompt = "A cat walking in the garden"
    frames = pipeline.generate(
        prompt=prompt,
        num_frames=49,
        height=480,
        width=720,
        guidance_scale=6.0
    )

    print(f"✅ 生成成功！生成了 {len(frames)} 帧")

if __name__ == "__main__":
    test_wan_pipeline()
```

#### 3.2 验证步骤

1. **单模型测试：** 先测试单个Wan模型能否正常加载和生成
2. **双模型加载测试：** 测试能否同时加载14B和1.3B模型
3. **协同推理测试：** 测试模型切换是否正常工作
4. **生成质量测试：** 对比生成视频质量

---

## 📝 详细修改清单

### 必须修改的文件（按优先级）

#### 🔴 高优先级（核心功能）

1. **`compression/hybrid_sd/inference_pipeline.py`**
   - [ ] Line 31: 修改导入 `HybridWanPipeline`
   - [ ] Line 688-690: 修改模型导入
   - [ ] Line 704-710: 修改VAE加载
   - [ ] Line 717-722: 修改Transformer加载
   - [ ] Line 748-755: 修改Base Pipeline创建
   - [ ] Line 759-765: 修改Hybrid Pipeline创建
   - [ ] Line 774-779: 修改额外Transformer加载

2. **`compression/hybrid_sd/diffusers/pipeline_cogvideox.py`**
   - [ ] 重命名为 `pipeline_wan.py`（可选）
   - [ ] 验证与WanPipeline的兼容性
   - [ ] 检查所有方法是否需要调整

#### 🟡 中优先级（测试和配置）

3. **`scripts/test_hybrid_video_inference_pipeline.py`**
   - [ ] 修改模型路径
   - [ ] 修改步数配置
   - [ ] 更新测试用例

4. **创建新文件 `scripts/test_hybrid_wan_pipeline.py`**
   - [ ] 编写Wan专用测试脚本

#### 🟢 低优先级（文档和清理）

5. **文档更新**
   - [ ] 更新README
   - [ ] 更新注释
   - [ ] 添加Wan使用说明

---

## ⚠️ 注意事项

### 1. 模型兼容性
- **Text Encoder:** Wan和CogVideoX都使用T5，无需修改
- **VAE:** 需要使用Wan专用的VAE（`AutoencoderKLWan`）
- **Transformer:** 架构可能不同，需要测试

### 2. 参数配置
- **步数分配:** Wan模型可能需要不同的步数配置
- **Guidance Scale:** 可能需要调整
- **分辨率:** Wan支持的分辨率可能不同

### 3. 内存管理
- Wan2.1-14B 比 CogVideoX-5B 更大，需要更多显存
- 可能需要调整VAE的slicing和tiling设置
- 考虑使用多GPU分配（VAE放在单独的GPU）

### 4. Scheduler配置
- 检查Wan是否需要特定的scheduler
- 验证 `snr_shift_scale` 等参数

---

## 🚀 实施步骤

### Step 1: 准备工作
1. 备份当前代码
2. 确认Wan模型已下载到正确路径
3. 检查diffusers版本是否支持Wan

### Step 2: 核心代码修改
1. 修改 `inference_pipeline.py`
2. 验证 `pipeline_cogvideox.py`（已经是HybridWanPipeline）
3. 运行语法检查

### Step 3: 测试验证
1. 单模型测试
2. 双模型加载测试
3. 协同推理测试
4. 生成质量对比

### Step 4: 优化调整
1. 调整步数配置
2. 优化内存使用
3. 性能测试

---

## 📊 预期结果

### 成功标准
- ✅ 能够成功加载Wan2.1-14B和1.3B模型
- ✅ 协同推理正常工作（模型切换无错误）
- ✅ 生成视频质量符合预期
- ✅ 内存使用在可接受范围内
- ✅ 推理速度有提升（相比单独使用14B）

### 性能指标
- **推理时间:** 应该比单独使用14B快
- **显存占用:** 应该在可用范围内
- **视频质量:** 应该接近或达到14B的质量

---

## 🔧 故障排查

### 常见问题

1. **模型加载失败**
   - 检查模型路径是否正确
   - 检查diffusers版本
   - 验证模型文件完整性

2. **显存不足**
   - 启用VAE slicing和tiling
   - 使用多GPU分配
   - 减少batch size

3. **生成质量差**
   - 调整步数配置
   - 调整guidance scale
   - 检查模型切换点

4. **模型切换错误**
   - 检查step_config配置
   - 验证transformer列表
   - 检查scheduler配置

---

## 📚 参考资料

- Wan官方文档
- CogVideoX文档（对比参考）
- Diffusers库文档
- Hybrid-SD原始论文

---

**创建时间:** 2026-02-08
**作者:** Claude
**版本:** 1.0
