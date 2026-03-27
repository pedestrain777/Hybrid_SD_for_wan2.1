# Hybrid-sd_wan 迁移计划（完整版 V3）：从 CogVideoX 到 Wan 系列

## ⚠️ 重要发现：关键差异

经过深入分析和实际验证，发现 **Wan 和 CogVideoX 有重大架构差异**，不仅仅是简单的模型替换！

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
| **transformer_2** | 不支持 | **支持双transformer架构** | ⚠️ Wan核心特性 |

---

## 🆕 新增：版本要求和环境检查

### 必需的版本约束

```python
# 在 inference_pipeline.py 文件开头添加版本检查
import diffusers
import transformers
import torch

# 版本检查
assert diffusers.__version__ >= "0.29.0", \
    f"需要 diffusers>=0.29.0 支持Wan，当前版本: {diffusers.__version__}"
assert transformers.__version__ >= "4.37.0", \
    f"需要 transformers>=4.37.0 支持UMT5，当前版本: {transformers.__version__}"
assert torch.__version__ >= "2.1.0", \
    f"需要 torch>=2.1.0，当前版本: {torch.__version__}"

print(f"✅ 环境检查通过:")
print(f"   - diffusers: {diffusers.__version__}")
print(f"   - transformers: {transformers.__version__}")
print(f"   - torch: {torch.__version__}")
```

### requirements.txt 更新

```txt
# Wan系列必需依赖
diffusers>=0.29.0
transformers>=4.37.0
torch>=2.1.0
accelerate>=0.20.0
safetensors>=0.3.1
```

---

## 📋 详细修改清单（增强版）

### 🔴 第一部分：`inference_pipeline.py` 修改

#### 0. **新增：文件开头版本检查**

```python
# 在所有导入之后，类定义之前添加
import diffusers
import transformers
import torch

# 版本检查
assert diffusers.__version__ >= "0.29.0", \
    f"需要 diffusers>=0.29.0 支持Wan，当前版本: {diffusers.__version__}"
assert transformers.__version__ >= "4.37.0", \
    f"需要 transformers>=4.37.0 支持UMT5，当前版本: {transformers.__version__}"
assert torch.__version__ >= "2.1.0", \
    f"需要 torch>=2.1.0，当前版本: {torch.__version__}"
```

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

#### 2. **增强：Text Encoder加载（Line 693-700）**

```python
# ❌ 修改前：
print(f"Loading Text Encoder from {self.weight_folders[0]}")
text_encoder = T5EncoderModel.from_pretrained(
    self.weight_folders[0], subfolder="text_encoder"
).to(self.device, dtype=torch.float16).requires_grad_(False)

tokenizer = T5Tokenizer.from_pretrained(
    self.weight_folders[0], subfolder="tokenizer"
)

# ✅ 修改后（增强版，带异常处理）：
print(f"Loading Text Encoder (UMT5) from {self.weight_folders[0]}")
try:
    text_encoder = UMT5EncoderModel.from_pretrained(
        self.weight_folders[0],
        subfolder="text_encoder"
    ).to(self.device, dtype=torch.float16).requires_grad_(False)
    print(f"✅ UMT5 Text Encoder 加载成功")
except Exception as e:
    print(f"❌ Text Encoder加载失败: {e}")
    raise

try:
    tokenizer = AutoTokenizer.from_pretrained(
        self.weight_folders[0],
        subfolder="tokenizer"
    )
    print(f"✅ Tokenizer 加载成功")
except Exception as e:
    print(f"❌ Tokenizer加载失败: {e}")
    raise
```

#### 3. **增强：VAE加载（Line 703-710）**

```python
# ❌ 修改前：
vae = AutoencoderKLCogVideoX.from_pretrained(
    self.weight_folders[0], subfolder="vae"
).to(self.vae_device, dtype=torch.float16).requires_grad_(False)

# ✅ 修改后（增强版，带异常处理和优化选项）：
try:
    vae = AutoencoderKLWan.from_pretrained(
        self.weight_folders[0],
        subfolder="vae"
    ).to(self.vae_device, dtype=torch.float16).requires_grad_(False)

    # 启用VAE优化（减少显存占用）
    vae.enable_slicing()
    vae.enable_tiling()

    print(f"✅ AutoencoderKLWan 加载成功 (device: {self.vae_device})")
    print(f"   - VAE slicing: 已启用")
    print(f"   - VAE tiling: 已启用")
except Exception as e:
    print(f"❌ VAE加载失败: {e}")
    raise
```

#### 4. **增强：Transformer加载（Line 717-722）**

```python
# ❌ 修改前：
transformer = CogVideoXTransformer3DModel.from_pretrained(
    path, subfolder="transformer", torch_dtype=torch.float16
).to(self.device).requires_grad_(False)

# ✅ 修改后（增强版，支持多GPU和device_map）：
try:
    # Wan2.1-14B 非常大，使用 device_map="auto" 自动分配到多GPU
    transformer = WanTransformer3DModel.from_pretrained(
        path,
        subfolder="transformer",
        torch_dtype=torch.float16,
        device_map="auto"  # 自动多GPU分配（14B模型必需）
    ).requires_grad_(False)

    print(f"✅ WanTransformer3DModel 加载成功")
    print(f"   - 模型路径: {path}")
    print(f"   - Device map: auto (多GPU自动分配)")
except Exception as e:
    print(f"❌ Transformer加载失败: {e}")
    print(f"   尝试使用单GPU加载...")
    try:
        transformer = WanTransformer3DModel.from_pretrained(
            path,
            subfolder="transformer",
            torch_dtype=torch.float16
        ).to(self.device).requires_grad_(False)
        print(f"✅ Transformer 单GPU加载成功")
    except Exception as e2:
        print(f"❌ 单GPU加载也失败: {e2}")
        raise
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
print(f"✅ WanPipeline 基础管道创建成功")
```

#### 6. **增强：Hybrid Pipeline创建（Line 759-765）**

```python
# ❌ 修改前：
self.pipe = HybridCogVideoXPipeline(
    transformer=base_pipeline.transformer,
    vae=base_pipeline.vae,
    text_encoder=base_pipeline.text_encoder,
    tokenizer=base_pipeline.tokenizer,
    scheduler=base_pipeline.scheduler
)

# ✅ 修改后（完整版，包含Wan特有参数）：
self.pipe = HybridWanPipeline(
    transformer=base_pipeline.transformer,
    vae=base_pipeline.vae,
    text_encoder=base_pipeline.text_encoder,
    tokenizer=base_pipeline.tokenizer,
    scheduler=base_pipeline.scheduler,
    # Wan特有参数（已验证存在于WanPipeline.__init__）
    transformer_2=None,        # 第二个transformer（用于双transformer架构）
    boundary_ratio=None,       # 边界比例（默认None，让Wan自动处理）
    expand_timesteps=False     # 是否扩展timesteps（默认False）
)
print(f"✅ HybridWanPipeline 创建成功")
print(f"   - transformer_2: {self.pipe.transformer_2 is not None}")
print(f"   - boundary_ratio: {self.pipe.boundary_ratio}")
print(f"   - expand_timesteps: {self.pipe.expand_timesteps}")
```


#### 7. **增强：额外Transformer加载（Line 774-779）**

```python
# ❌ 修改前：
additional_transformer = CogVideoXTransformer3DModel.from_pretrained(
    self.weight_folders[i],
    subfolder="transformer",
    torch_dtype=torch.float16
).to(self.device).requires_grad_(False)

# ✅ 修改后（增强版，支持多GPU）：
try:
    additional_transformer = WanTransformer3DModel.from_pretrained(
        self.weight_folders[i],
        subfolder="transformer",
        torch_dtype=torch.float16,
        device_map="auto"  # 多GPU自动分配
    ).requires_grad_(False)
    print(f"✅ 额外 Transformer #{i} 加载成功")
except Exception as e:
    print(f"❌ 额外 Transformer #{i} 加载失败: {e}")
    # 尝试单GPU加载
    additional_transformer = WanTransformer3DModel.from_pretrained(
        self.weight_folders[i],
        subfolder="transformer",
        torch_dtype=torch.float16
    ).to(self.device).requires_grad_(False)
    print(f"✅ 额外 Transformer #{i} 单GPU加载成功")
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
    print(f"✅ 使用自定义 FlowMatchEulerDiscreteScheduler")
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

#### 3. **增强：Scheduler类型检查修改（Line 514-521, 594, 679-695）**

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

#### 4. **关键：scheduler.step() 调用修改（Line 679-695）**

**重要：** Wan使用Flow Matching而不是DDPM/DDIM，scheduler.step()的调用方式不同：

```python
# ❌ 修改前（CogVideoX DDPM-based）：
latents = self.scheduler.step(
    model_output,
    timestep,
    latents,
    **extra_step_kwargs
).prev_sample

# ✅ 修改后（Wan Flow Matching）：
# FlowMatchEulerDiscreteScheduler.step() 参数签名：
# step(model_output, timestep, sample, s_churn=0.0, s_tmin=0.0, 
#      s_tmax=inf, s_noise=1.0, generator=None, return_dict=True)

latents = self.scheduler.step(
    model_output,
    timestep,
    latents,
    s_churn=0.0,      # 随机性控制（Flow Matching特有）
    s_noise=1.0,      # 噪声强度（Flow Matching特有）
    return_dict=False
)[0]

# 注意：
# 1. 没有 eta 参数（那是DDIM的参数）
# 2. 有 s_churn, s_noise 等Flow Matching特有参数
# 3. return_dict=False 返回tuple，取[0]获取latents
```

#### 5. Pipeline Output修改（Line 821-823）

```python
# ❌ 修改前：
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
return CogVideoXPipelineOutput(frames=video)

# ✅ 修改后：
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
return WanPipelineOutput(frames=video)
```

#### 6. DDIM Scheduler修改（Line 583-588）

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

## 🎯 Wan特有特性详解（已验证）

### 1. **transformer_2 支持（已验证存在）**

通过检查 `WanPipeline.__init__` 签名，确认Wan支持双transformer架构：

```python
WanPipeline.__init__(
    tokenizer: AutoTokenizer,
    text_encoder: UMT5EncoderModel,
    vae: AutoencoderKLWan,
    scheduler: FlowMatchEulerDiscreteScheduler,
    transformer: Optional[WanTransformer3DModel] = None,
    transformer_2: Optional[WanTransformer3DModel] = None,  # ✅ 确认存在
    boundary_ratio: Optional[float] = None,                  # ✅ 确认存在
    expand_timesteps: bool = False                           # ✅ 确认存在
)
```

**在HybridWanPipeline中的使用：**

```python
def __init__(self, ..., transformer_2=None, boundary_ratio=None, expand_timesteps=False):
    super().__init__(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        transformer_2=transformer_2,      # Wan特有：第二个transformer
        boundary_ratio=boundary_ratio,    # Wan特有：边界比例（0.3-0.7推荐）
        expand_timesteps=expand_timesteps # Wan特有：是否扩展timesteps
    )
```

**transformer_2 的作用：**
- 类似SDXL的双UNet架构
- 可以用14B作为transformer，1.3B作为transformer_2
- boundary_ratio控制两个transformer的切换点

### 2. **Flow Matching调度（已验证参数）**

通过检查 `FlowMatchEulerDiscreteScheduler.step` 签名，确认参数：

```python
FlowMatchEulerDiscreteScheduler.step(
    model_output: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    sample: torch.FloatTensor,
    s_churn: float = 0.0,           # ✅ 随机性控制
    s_tmin: float = 0.0,            # ✅ 最小时间步
    s_tmax: float = inf,            # ✅ 最大时间步
    s_noise: float = 1.0,           # ✅ 噪声强度
    generator: Optional[Generator] = None,
    per_token_timesteps: Optional[torch.Tensor] = None,
    return_dict: bool = True
) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]
```

**关键差异：**

| 特性 | DDPM/DDIM (CogVideoX) | Flow Matching (Wan) |
|------|----------------------|---------------------|
| Timestep范围 | [0, 1000] | [0, 1] |
| 调度方式 | 离散去噪步骤 | 连续流 |
| 参数 | eta, variance_noise | s_churn, s_noise |
| 返回值 | .prev_sample | tuple[0] |

### 3. **UMT5 Text Encoder**

Wan使用UMT5（Unified Multilingual T5），支持多语言：

```python
# UMT5特点：
# - 支持100+语言
# - 更好的多语言prompt理解
# - 可能需要不同的prompt格式化
```

---

## 📝 完整修改文件列表（增强版）

### 必须修改的文件

1. **`compression/hybrid_sd/inference_pipeline.py`**
   - [ ] **新增**: 文件开头添加版本检查
   - [ ] Line 31: 修改导入 `HybridWanPipeline`
   - [ ] Line 688-690: 修改导入（UMT5, AutoTokenizer, Wan组件）
   - [ ] Line 693-700: 修改Text Encoder和Tokenizer加载（增强：异常处理）
   - [ ] Line 704-710: 修改VAE加载（增强：slicing/tiling优化）
   - [ ] Line 717-722: 修改Transformer加载（增强：device_map="auto"）
   - [ ] Line 748-755: 修改Base Pipeline创建
   - [ ] Line 759-765: 修改Hybrid Pipeline创建（增强：transformer_2等参数）
   - [ ] Line 774-779: 修改额外Transformer加载（增强：多GPU支持）
   - [ ] Line 810-814: 修改Scheduler配置（可选）

2. **`compression/hybrid_sd/diffusers/pipeline_cogvideox.py`**
   - [ ] 重命名为 `pipeline_wan.py`
   - [ ] Line 32: 删除 `retrieve_timesteps` 导入
   - [ ] Line 410: 修改timesteps获取方式
   - [ ] Line 514-521: 修改Scheduler类型检查
   - [ ] Line 583-588: 修改DDIM Scheduler处理
   - [ ] Line 594: 修改DPM Scheduler检查
   - [ ] Line 679-695: **关键修改** scheduler.step()调用（s_churn, s_noise参数）
   - [ ] Line 821-823: 修改Pipeline Output

3. **`scripts/test_hybrid_video_inference_pipeline.py`**
   - [ ] 修改模型路径
   - [ ] 修改步数配置（15-30步）
   - [ ] 更新测试用例

4. **新增文件：`requirements_wan.txt`**
   ```txt
   diffusers>=0.29.0
   transformers>=4.37.0
   torch>=2.1.0
   accelerate>=0.20.0
   safetensors>=0.3.1
   ```

---


## ⚠️ 关键注意事项（增强版）

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
latents = scheduler.step(
    model_output, t, latents, 
    s_churn=0.0, s_noise=1.0, 
    return_dict=False
)[0]
```

### 2. **Text Encoder差异**
- **T5**: 英文为主
- **UMT5**: 多语言支持，可能需要不同的prompt处理

### 3. **模型大小和显存管理**
- Wan2.1-14B 比 CogVideoX-5B 大得多（约3倍）
- **必须使用 device_map="auto"** 进行多GPU分配
- **必须启用 VAE slicing 和 tiling**
- 考虑使用 gradient checkpointing（如果显存仍不足）

### 4. **步数配置**
Flow Matching通常需要更少的步数：
```python
# CogVideoX: 25-50步
args.steps = [10, 15]  # 总共25步

# Wan: 15-30步（Flow Matching更高效）
args.steps = [8, 12]  # 总共20步
# 或
args.steps = [10, 15]  # 总共25步
```

### 5. **推理参数调整**
```python
# CogVideoX推荐参数：
guidance_scale = 6.0
num_inference_steps = 50

# Wan推荐参数：
guidance_scale = 7.5-10.0  # 更高的guidance
num_inference_steps = 20-30  # 更少的步数
```

---

## 🧪 测试策略（增强版）

### 阶段0：环境验证
```python
# 创建 test_environment.py
import diffusers
import transformers
import torch

print(f"diffusers: {diffusers.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"torch: {torch.__version__}")

# 测试导入
from diffusers import WanPipeline, WanTransformer3DModel, AutoencoderKLWan
from transformers import UMT5EncoderModel, AutoTokenizer
print("✅ 所有组件导入成功")
```

### 阶段1：单组件测试
```python
# test_wan_components.py
from transformers import UMT5EncoderModel, AutoTokenizer
from diffusers import AutoencoderKLWan, WanTransformer3DModel

model_path = "/path/to/Wan2.1-T2V-14B-Diffusers"

# 测试1: Text Encoder
print("测试 UMT5 Text Encoder...")
text_encoder = UMT5EncoderModel.from_pretrained(
    model_path, subfolder="text_encoder"
).to("cuda", dtype=torch.float16)
print("✅ UMT5 加载成功")

# 测试2: Tokenizer
print("测试 AutoTokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path, subfolder="tokenizer"
)
print("✅ Tokenizer 加载成功")

# 测试3: VAE
print("测试 AutoencoderKLWan...")
vae = AutoencoderKLWan.from_pretrained(
    model_path, subfolder="vae"
).to("cuda", dtype=torch.float16)
vae.enable_slicing()
vae.enable_tiling()
print("✅ VAE 加载成功")

# 测试4: Transformer
print("测试 WanTransformer3DModel...")
transformer = WanTransformer3DModel.from_pretrained(
    model_path, subfolder="transformer",
    torch_dtype=torch.float16,
    device_map="auto"
)
print("✅ Transformer 加载成功")
```

### 阶段2：单模型Pipeline测试
```python
# test_wan_single.py
from diffusers import WanPipeline
import torch

model_path = "/path/to/Wan2.1-T2V-14B-Diffusers"

print("加载 WanPipeline...")
pipe = WanPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16
)
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
pipe.to("cuda")

print("生成测试视频...")
video = pipe(
    prompt="A cat walking in the garden",
    num_frames=49,
    height=480,
    width=720,
    num_inference_steps=20,
    guidance_scale=7.5
).frames

print(f"✅ 生成成功！生成了 {len(video)} 帧")
```

### 阶段3：Hybrid推理测试
```python
# test_hybrid_wan.py
from compression.hybrid_sd.inference_pipeline import HybridVideoInferencePipeline

class MockArgs:
    def __init__(self, steps):
        self.steps = steps
        self.enable_xformers_memory_efficient_attention = False
        self.use_custom_scheduler = False
        self.vae_device = None

model_path_14b = "/path/to/Wan2.1-T2V-14B-Diffusers"
model_path_1_3b = "/path/to/Wan2.1-T2V-1.3B-Diffusers"

weight_folders = [model_path_14b, model_path_1_3b]
args = MockArgs(steps=[10, 15])  # 14B用10步，1.3B用15步

print("初始化 Hybrid Pipeline...")
pipeline = HybridVideoInferencePipeline(
    weight_folders=weight_folders,
    seed=42,
    device="cuda",
    args=args
)

print("加载模型...")
pipeline.set_pipe_and_generator()

print("生成视频...")
frames = pipeline.generate(
    prompt="A cat walking in the garden",
    num_frames=49,
    height=480,
    width=720,
    guidance_scale=7.5
)

print(f"✅ Hybrid生成成功！生成了 {len(frames)} 帧")
```

### 阶段4：生成质量对比测试
```python
# 对比不同配置的生成质量
test_configs = [
    {"steps": [8, 12], "guidance": 7.5},
    {"steps": [10, 15], "guidance": 7.5},
    {"steps": [10, 15], "guidance": 10.0},
]

for config in test_configs:
    print(f"测试配置: {config}")
    # 生成并保存视频
    # 对比质量
```

---

## 🚀 实施步骤（详细版）

### Step 1: 环境准备和备份

```bash
# 1. 备份代码
cd /data/chenjiayu/minyu_lee
cp -r Hybrid-sd_wan Hybrid-sd_wan_backup_$(date +%Y%m%d)

# 2. 检查环境
cd Hybrid-sd_wan
python -c "import diffusers; print(f'diffusers: {diffusers.__version__}')"
python -c "from diffusers import WanPipeline; print('✅ WanPipeline可用')"

# 3. 确认模型已下载
ls -lh /path/to/Wan2.1-T2V-14B-Diffusers
ls -lh /path/to/Wan2.1-T2V-1.3B-Diffusers
```

### Step 2: 代码修改

```bash
# 1. 重命名pipeline文件
cd compression/hybrid_sd/diffusers
mv pipeline_cogvideox.py pipeline_wan.py

# 2. 创建测试脚本目录
cd /data/chenjiayu/minyu_lee/Hybrid-sd_wan
mkdir -p tests/wan_migration

# 3. 按照修改清单逐一修改文件
# - inference_pipeline.py
# - pipeline_wan.py
# - test_hybrid_video_inference_pipeline.py
```

### Step 3: 逐步测试

```bash
# 测试1: 环境验证
python tests/wan_migration/test_environment.py

# 测试2: 单组件测试
python tests/wan_migration/test_wan_components.py

# 测试3: 单模型测试
python tests/wan_migration/test_wan_single.py

# 测试4: Hybrid测试
python tests/wan_migration/test_hybrid_wan.py
```

### Step 4: 性能优化

```bash
# 如果显存不足，尝试：
# 1. 启用gradient checkpointing
# 2. 减少batch size
# 3. 使用更激进的VAE tiling
# 4. 考虑使用量化（int8）
```

---

## 📊 预期挑战和解决方案（增强版）

### 挑战1：Flow Matching vs DDPM ⚠️ 最关键
**问题**: scheduler.step()调用方式完全不同
**解决**: 
- 使用 `s_churn=0.0, s_noise=1.0` 参数
- 使用 `return_dict=False` 并取 `[0]`
- 移除所有 `eta` 参数引用

### 挑战2：显存不足 ⚠️ 很可能遇到
**问题**: Wan2.1-14B 需要约28GB显存
**解决**:
```python
# 方案1: 多GPU分配
transformer = WanTransformer3DModel.from_pretrained(
    path, device_map="auto"  # 自动分配到多GPU
)

# 方案2: VAE优化
vae.enable_slicing()
vae.enable_tiling()

# 方案3: Gradient checkpointing
transformer.enable_gradient_checkpointing()

# 方案4: 使用更小的分辨率
height, width = 480, 720  # 而不是 720, 1280
```

### 挑战3：UMT5 vs T5
**问题**: Text Encoder不同，可能影响prompt理解
**解决**: 
- 测试不同prompt格式
- 必要时添加prompt预处理
- 利用UMT5的多语言能力

### 挑战4：步数配置
**问题**: Flow Matching的最优步数未知
**解决**: 
- 从20步开始测试
- 逐步调整到15-30步范围
- 对比生成质量和速度

### 挑战5：transformer_2 使用
**问题**: 如何正确使用双transformer架构
**解决**:
```python
# 方案1: 不使用transformer_2（先跑通基础功能）
transformer_2=None

# 方案2: 使用transformer_2（优化阶段）
# 14B作为主transformer，1.3B作为transformer_2
# boundary_ratio=0.5 表示在50%的步数切换
```

---

## 📚 参考资料

- [Wan官方文档](https://huggingface.co/Wan-AI)
- [Flow Matching论文](https://arxiv.org/abs/2210.02747)
- [UMT5文档](https://huggingface.co/docs/transformers/model_doc/umt5)
- [Diffusers Wan Pipeline源码](https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/wan)
- [FlowMatchEulerDiscreteScheduler文档](https://huggingface.co/docs/diffusers/api/schedulers/flow_match_euler_discrete)

---

## 📋 实施检查清单

### 代码修改
- [ ] 添加版本检查代码
- [ ] 修改 inference_pipeline.py (9处修改)
- [ ] 修改 pipeline_wan.py (7处修改)
- [ ] 更新测试脚本
- [ ] 创建 requirements_wan.txt

### 测试验证
- [ ] 环境验证测试通过
- [ ] 单组件测试通过
- [ ] 单模型Pipeline测试通过
- [ ] Hybrid推理测试通过
- [ ] 生成质量达到预期

### 优化调整
- [ ] 步数配置优化
- [ ] 显存使用优化
- [ ] 推理速度测试
- [ ] 多种prompt测试

### 文档更新
- [ ] 更新README
- [ ] 添加Wan使用说明
- [ ] 记录最佳实践
- [ ] 添加故障排查指南

---

**创建时间:** 2026-02-08  
**作者:** Claude  
**版本:** 3.0 (增强完整版)  
**状态:** 包含所有验证过的差异、增强的鲁棒性、Wan特有特性详解

**主要改进:**
- ✅ 验证了 transformer_2, boundary_ratio, expand_timesteps 参数
- ✅ 验证了 FlowMatchEulerDiscreteScheduler.step() 参数签名
- ✅ 添加了版本检查和异常处理
- ✅ 添加了多GPU支持 (device_map="auto")
- ✅ 添加了VAE优化 (slicing/tiling)
- ✅ 提供了完整的测试策略
- ✅ 详细的故障排查方案

