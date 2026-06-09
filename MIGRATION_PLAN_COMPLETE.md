# Hybrid-sd_wan 迁移计划（完整版）：从 CogVideoX 到 Wan 系列

## ⚠️ 重要发现：关键差异

经过深入分析，发现 **Wan 和 CogVideoX 有重大架构差异**，不仅仅是简单的模型替换！

### 🔴 核心差异对比表

| 组件 | CogVideoX | Wan | 影响 |
|------|-----------|-----|------|
| **Text Encoder** | `T5EncoderModel` | `UMT5EncoderModel` | ⚠️ 必须修改 |
| **VAE** | `AutoencoderKLCogVideoX` | `AutoencoderKLWan` | ⚠️ 必须修改 |
| **Transformer** | `CogVideoXTransformer3DModel` | `WanTransformer3DModel` | ⚠️ 必须修改 |
| **Scheduler** | `CogVideoXDPMScheduler` / `CogVideoXDDIMScheduler` | `FlowMatchEulerDiscreteScheduler` | ⚠️ 必须修改 |
| **Pipeline Output** | `CogVideoXPipelineOutput` | `WanPipelineOutput` | ⚠️ 必须修改 |
| **retrieve_timesteps** | 有专用函数 | 无专用函数（使用通用方法） | ⚠️ 需要适配 |
| **Tokenizer** | `T5Tokenizer` | `AutoTokenizer` | ⚠️ 必须修改 |

---

## 📋 详细修改清单

### 🔴 第一部分：`inference_pipeline.py` 修改

#### 1. 导入语句修改（Line 688-690）

```python
# ❌ 修改前（CogVideoX）：
from transformers import T5EncoderModel, T5Tokenizer
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
from diffusers import CogVideoXPipeline

# ✅ 修改后（Wan）：
from transformers import UMT5EncoderModel, AutoTokenizer
from diffusers import AutoencoderKLWan, WanTransformer3DModel
from diffusers import WanPipeline
```

#### 2. Text Encoder加载（Line 693-700）

```python
# ❌ 修改前：
print(f"Loading Text Encoder from {self.weight_folders[0]}")
text_encoder = T5EncoderModel.from_pretrained(
    self.weight_folders[0], subfolder="text_encoder"
).to(self.device, dtype=torch.float16).requires_grad_(False)

tokenizer = T5Tokenizer.from_pretrained(
    self.weight_folders[0], subfolder="tokenizer"
)

# ✅ 修改后：
print(f"Loading Text Encoder (UMT5) from {self.weight_folders[0]}")
text_encoder = UMT5EncoderModel.from_pretrained(
    self.weight_folders[0], subfolder="text_encoder"
).to(self.device, dtype=torch.float16).requires_grad_(False)

tokenizer = AutoTokenizer.from_pretrained(
    self.weight_folders[0], subfolder="tokenizer"
)
```

#### 3. VAE加载（Line 703-710）

```python
# ❌ 修改前：
vae = AutoencoderKLCogVideoX.from_pretrained(
    self.weight_folders[0], subfolder="vae"
).to(self.vae_device, dtype=torch.float16).requires_grad_(False)

# ✅ 修改后：
vae = AutoencoderKLWan.from_pretrained(
    self.weight_folders[0], subfolder="vae"
).to(self.vae_device, dtype=torch.float16).requires_grad_(False)
```

#### 4. Transformer加载（Line 717-722）

```python
# ❌ 修改前：
transformer = CogVideoXTransformer3DModel.from_pretrained(
    path, subfolder="transformer", torch_dtype=torch.float16
).to(self.device).requires_grad_(False)

# ✅ 修改后：
transformer = WanTransformer3DModel.from_pretrained(
    path, subfolder="transformer", torch_dtype=torch.float16
).to(self.device).requires_grad_(False)
```

#### 5. Base Pipeline创建（Line 748-755）

```python
# ❌ 修改前：
base_pipeline = CogVideoXPipeline.from_pretrained(
    self.weight_folders[0],
    text_encoder=text_encoder,
    vae=vae,
    tokenizer=tokenizer,
    transformer=transformers[0],
    torch_dtype=torch.float16
)

# ✅ 修改后：
base_pipeline = WanPipeline.from_pretrained(
    self.weight_folders[0],
    text_encoder=text_encoder,
    vae=vae,
    tokenizer=tokenizer,
    transformer=transformers[0],
    torch_dtype=torch.float16
)
```

#### 6. Hybrid Pipeline创建（Line 759-765）

```python
# ❌ 修改前：
self.pipe = HybridCogVideoXPipeline(
    transformer=base_pipeline.transformer,
    vae=base_pipeline.vae,
    text_encoder=base_pipeline.text_encoder,
    tokenizer=base_pipeline.tokenizer,
    scheduler=base_pipeline.scheduler
)

# ✅ 修改后：
self.pipe = HybridWanPipeline(
    transformer=base_pipeline.transformer,
    vae=base_pipeline.vae,
    text_encoder=base_pipeline.text_encoder,
    tokenizer=base_pipeline.tokenizer,
    scheduler=base_pipeline.scheduler,
    # Wan特有参数（如果需要）
    transformer_2=None,  # 可选的第二个transformer
    boundary_ratio=None,  # 边界比例
    expand_timesteps=False  # 是否扩展timesteps
)
```

#### 7. 额外Transformer加载（Line 774-779）

```python
# ❌ 修改前：
additional_transformer = CogVideoXTransformer3DModel.from_pretrained(
    self.weight_folders[i],
    subfolder="transformer",
    torch_dtype=torch.float16
).to(self.device).requires_grad_(False)

# ✅ 修改后：
additional_transformer = WanTransformer3DModel.from_pretrained(
    self.weight_folders[i],
    subfolder="transformer",
    torch_dtype=torch.float16
).to(self.device).requires_grad_(False)
```

#### 8. Scheduler配置（Line 810-814）

```python
# ❌ 修改前：
if hasattr(self.args, 'use_dpm_solver') and self.args.use_dpm_solver:
    from diffusers import CogVideoXDPMScheduler
    self.pipe.scheduler = CogVideoXDPMScheduler.from_config(
        self.pipe.scheduler.config, timestep_spacing="trailing"
    )

# ✅ 修改后：
# Wan使用FlowMatchEulerDiscreteScheduler，通常不需要修改
# 如果需要自定义scheduler：
if hasattr(self.args, 'use_custom_scheduler') and self.args.use_custom_scheduler:
    from diffusers import FlowMatchEulerDiscreteScheduler
    self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        self.pipe.scheduler.config
    )
# 注意：Wan不使用DPM solver，使用Flow Matching
```

---

### 🔴 第二部分：`pipeline_cogvideox.py` 修改

#### 文件重命名
```bash
mv compression/hybrid_sd/diffusers/pipeline_cogvideox.py \
   compression/hybrid_sd/diffusers/pipeline_wan.py
```

#### 1. 导入语句修改（Line 31-33）

```python
# ❌ 修改前：
from diffusers import WanPipeline
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from diffusers.utils import logging

# ✅ 修改后：
from diffusers import WanPipeline
from diffusers.utils import logging
# Wan不需要retrieve_timesteps，使用scheduler的内置方法
```

#### 2. retrieve_timesteps调用修改（Line 410）

```python
# ❌ 修改前：
timesteps, num_inference_steps = retrieve_timesteps(
    self.scheduler,
    num_inference_steps,
    device,
    timesteps,
    sigmas,
)

# ✅ 修改后：
# Wan使用scheduler的set_timesteps方法
self.scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = self.scheduler.timesteps
# 如果提供了自定义timesteps
if timesteps is not None:
    self.scheduler.timesteps = timesteps
```

#### 3. Scheduler类型检查修改（Line 514-521, 594, 679-695）

```python
# ❌ 修改前：
from diffusers import CogVideoXDPMScheduler
if isinstance(self.scheduler, CogVideoXDPMScheduler):
    # CogVideoX特定逻辑
    ...

# ✅ 修改后：
from diffusers import FlowMatchEulerDiscreteScheduler
if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
    # Wan特定逻辑
    # Flow Matching使用不同的step方法
    ...
```

**重要：** Wan使用Flow Matching而不是DDPM/DDIM，scheduler.step()的调用方式可能不同：

```python
# CogVideoX (DDPM-based):
latents = self.scheduler.step(
    model_output,
    timestep,
    latents,
    **extra_step_kwargs
).prev_sample

# Wan (Flow Matching):
latents = self.scheduler.step(
    model_output,
    timestep,
    latents,
    return_dict=False
)[0]
```

#### 4. Pipeline Output修改（Line 821-823）

```python
# ❌ 修改前：
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
return CogVideoXPipelineOutput(frames=video)

# ✅ 修改后：
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
return WanPipelineOutput(frames=video)
```

#### 5. DDIM Scheduler修改（Line 583-588）

```python
# ❌ 修改前：
from diffusers import CogVideoXDDIMScheduler
if isinstance(self.scheduler, CogVideoXDDIMScheduler):
    self.scheduler = CogVideoXDDIMScheduler.from_config(updated_config)

# ✅ 修改后：
# Wan不使用DDIM，使用FlowMatchEulerDiscreteScheduler
# 如果需要更新scheduler配置：
from diffusers import FlowMatchEulerDiscreteScheduler
if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
    self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(updated_config)
```

---

### 🔴 第三部分：`inference_pipeline.py` 导入修改（Line 31）

```python
# ❌ 修改前：
from .diffusers.pipeline_cogvideox import HybridWanPipeline

# ✅ 修改后：
from .diffusers.pipeline_wan import HybridWanPipeline
```

---

## 🎯 Wan特有特性需要添加

### 1. **transformer_2 支持**
Wan支持双transformer架构（类似SDXL），需要在HybridWanPipeline中处理：

```python
def __init__(self, ..., transformer_2=None, boundary_ratio=None, expand_timesteps=False):
    super().__init__(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        transformer_2=transformer_2,  # Wan特有
        boundary_ratio=boundary_ratio,  # Wan特有
        expand_timesteps=expand_timesteps  # Wan特有
    )
```

### 2. **Flow Matching调度**
Wan使用Flow Matching而不是DDPM，需要理解其工作原理：

- **DDPM/DDIM**: 从噪声逐步去噪到图像
- **Flow Matching**: 学习从噪声到图像的连续流

```python
# Flow Matching的timestep通常是[0, 1]范围
# 而DDPM是[0, 1000]范围
```

### 3. **UMT5 Text Encoder**
Wan使用UMT5（Unified Multilingual T5），支持多语言：

```python
# UMT5可能需要不同的prompt处理
# 检查是否需要特殊的prompt清理或格式化
```

---

## 📝 完整修改文件列表

### 必须修改的文件

1. **`compression/hybrid_sd/inference_pipeline.py`**
   - [ ] Line 31: 修改导入 `HybridWanPipeline`
   - [ ] Line 688-690: 修改导入（UMT5, AutoTokenizer, Wan组件）
   - [ ] Line 693-700: 修改Text Encoder和Tokenizer加载
   - [ ] Line 704-710: 修改VAE加载
   - [ ] Line 717-722: 修改Transformer加载
   - [ ] Line 748-755: 修改Base Pipeline创建
   - [ ] Line 759-765: 修改Hybrid Pipeline创建（添加Wan特有参数）
   - [ ] Line 774-779: 修改额外Transformer加载
   - [ ] Line 810-814: 修改Scheduler配置（可选）

2. **`compression/hybrid_sd/diffusers/pipeline_cogvideox.py`**
   - [ ] 重命名为 `pipeline_wan.py`
   - [ ] Line 32: 删除 `retrieve_timesteps` 导入
   - [ ] Line 410: 修改timesteps获取方式
   - [ ] Line 514-521: 修改Scheduler类型检查
   - [ ] Line 583-588: 修改DDIM Scheduler处理
   - [ ] Line 594: 修改DPM Scheduler检查
   - [ ] Line 679-695: 修改scheduler.step()调用
   - [ ] Line 821-823: 修改Pipeline Output

3. **`scripts/test_hybrid_video_inference_pipeline.py`**
   - [ ] 修改模型路径
   - [ ] 修改步数配置
   - [ ] 更新测试用例

---

## ⚠️ 关键注意事项

### 1. **Scheduler差异（最重要！）**

**CogVideoX使用DDPM/DDIM:**
```python
# 噪声调度：t ∈ [0, 1000]
# 去噪过程：从高噪声到低噪声
latents = scheduler.step(model_output, t, latents).prev_sample
```

**Wan使用Flow Matching:**
```python
# 流调度：t ∈ [0, 1]
# 流过程：从噪声流向数据
latents = scheduler.step(model_output, t, latents, return_dict=False)[0]
```

### 2. **Text Encoder差异**
- **T5**: 英文为主
- **UMT5**: 多语言支持，可能需要不同的prompt处理

### 3. **模型大小**
- Wan2.1-14B 比 CogVideoX-5B 大得多
- 需要更多显存和更长的推理时间

### 4. **步数配置**
Flow Matching通常需要更少的步数：
```python
# CogVideoX: 25-50步
args.steps = [10, 15]  # 总共25步

# Wan: 可能只需要15-30步
args.steps = [8, 12]  # 总共20步（需要实验确定）
```

---

## 🧪 测试策略

### 阶段1：单模型测试
```python
# 测试单个Wan模型能否加载和生成
model_path = "/path/to/Wan2.1-T2V-14B-Diffusers"
# 不使用hybrid，直接用WanPipeline测试
```

### 阶段2：组件兼容性测试
```python
# 测试各个组件是否正确加载
# - UMT5 Text Encoder
# - AutoencoderKLWan
# - WanTransformer3DModel
# - FlowMatchEulerDiscreteScheduler
```

### 阶段3：Hybrid推理测试
```python
# 测试模型切换是否正常
# 测试不同步数配置
```

### 阶段4：生成质量测试
```python
# 对比生成视频质量
# 测试不同prompt
# 测试不同参数配置
```

---

## 🚀 实施步骤（更新）

### Step 1: 环境准备
```bash
# 1. 备份代码
cp -r Hybrid-sd_wan Hybrid-sd_wan_backup

# 2. 确认diffusers版本支持Wan
python -c "from diffusers import WanPipeline; print('OK')"

# 3. 确认模型已下载
ls /path/to/Wan2.1-T2V-14B-Diffusers
ls /path/to/Wan2.1-T2V-1.3B-Diffusers
```

### Step 2: 代码修改
```bash
# 1. 重命名pipeline文件
cd compression/hybrid_sd/diffusers
mv pipeline_cogvideox.py pipeline_wan.py

# 2. 修改inference_pipeline.py（按照上述清单）
# 3. 修改pipeline_wan.py（按照上述清单）
```

### Step 3: 单模型测试
```python
# 创建测试脚本test_wan_single.py
from diffusers import WanPipeline

pipe = WanPipeline.from_pretrained(
    "/path/to/Wan2.1-T2V-14B-Diffusers",
    torch_dtype=torch.float16
).to("cuda")

video = pipe(
    prompt="A cat walking in the garden",
    num_frames=49,
    height=480,
    width=720
).frames

print(f"Generated {len(video)} frames")
```

### Step 4: Hybrid测试
```python
# 使用修改后的HybridVideoInferencePipeline
# 测试双模型协同推理
```

---

## 📊 预期挑战和解决方案

### 挑战1：Flow Matching vs DDPM
**问题**: scheduler.step()调用方式不同
**解决**: 在HybridWanPipeline中适配Flow Matching的调用方式

### 挑战2：UMT5 vs T5
**问题**: Text Encoder不同，可能影响prompt理解
**解决**: 测试不同prompt格式，必要时添加prompt预处理

### 挑战3：显存不足
**问题**: Wan2.1-14B非常大
**解决**:
- 使用VAE slicing和tiling
- 多GPU分配（VAE放单独GPU）
- 考虑使用gradient checkpointing

### 挑战4：步数配置
**问题**: Flow Matching的最优步数未知
**解决**: 实验不同步数配置，找到质量和速度的平衡点

---

## 📚 参考资料

- [Wan官方文档](https://huggingface.co/Wan-AI)
- [Flow Matching论文](https://arxiv.org/abs/2210.02747)
- [UMT5文档](https://huggingface.co/docs/transformers/model_doc/umt5)
- [Diffusers Wan Pipeline源码](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/wan)

---

**创建时间:** 2026-02-08
**作者:** Claude
**版本:** 2.0 (完整版)
**状态:** 包含所有关键差异和Wan特有特性
