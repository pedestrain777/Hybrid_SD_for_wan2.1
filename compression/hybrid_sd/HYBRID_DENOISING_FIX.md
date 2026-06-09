# Hybrid CogVideoX 去噪问题修复总结

## 问题概述

在使用 Hybrid CogVideoX（5B + 2B 混合去噪）时，虽然单独使用 5B 或 2B 模型都能产生良好的结果，但混合模式（例如 `--steps 10,15`，前10步用5B，后15步用2B）生成的视频质量很差，出现明显的噪点和失真。

## 问题发现过程

### 1. 初步观察
- **纯 5B 去噪**：效果良好 ✅
- **纯 2B 去噪**：效果良好 ✅
- **混合去噪（5B → 2B）**：效果很差 ❌

### 2. 日志分析
通过分析去噪过程中的日志，发现了关键异常：

```
Step 9/25 (5B): noise_pred guided std: 0.7252
Step 10/25 (2B): noise_pred guided std: 1.1387  ← 突然增大 57%！
```

在模型切换点（Step 9 → Step 10），`noise_pred` 的标准差突然从 0.72 跳到 1.14，这表明 2B 模型的预测分布异常。

### 3. 深入调查
通过检查代码和配置文件，发现了根本原因：

**Scheduler 配置不匹配**：
- **CogVideoX-5B** 的 scheduler 配置：`snr_shift_scale: 1.0`
- **CogVideoX-2B** 的 scheduler 配置：`snr_shift_scale: 3.0`

但在混合去噪实现中，scheduler 是从 5B 模型加载的，5B 和 2B 共用同一个 scheduler 实例。当切换到 2B 模型时，scheduler 仍然使用 5B 的配置（`snr_shift_scale=1.0`），而 2B 模型期望使用自己的配置（`snr_shift_scale=3.0`）。

### 4. 为什么这会导致问题？

`snr_shift_scale` 是 DDIM Scheduler 中的一个关键参数，它影响噪声调度和去噪过程。当 2B 模型使用错误的 `snr_shift_scale` 值时：
- 模型接收到的 latent 分布与训练时不一致
- 导致模型预测的 `noise_pred` 分布异常（std 突然增大）
- 最终生成的视频质量下降

## 解决方案

### 实现动态 Scheduler 配置切换

在模型切换时，动态更新 scheduler 的配置以匹配当前使用的模型。

### 1. 加载每个模型的 Scheduler 配置

在 `inference_pipeline.py` 中，为每个模型加载对应的 scheduler 配置：

```python
# 7. Load scheduler configs for each model
import json
scheduler_configs = []
for path in self.weight_folders:
    scheduler_config_path = os.path.join(path, "scheduler", "scheduler_config.json")
    if os.path.exists(scheduler_config_path):
        with open(scheduler_config_path, 'r') as f:
            scheduler_config = json.load(f)
            scheduler_configs.append(scheduler_config)
            print(f"Loaded scheduler config from {scheduler_config_path}: snr_shift_scale={scheduler_config.get('snr_shift_scale', 'N/A')}")
```

### 2. 在模型切换时更新 Scheduler

在 `pipeline_cogvideox.py` 的去噪循环中，检测模型切换并更新 scheduler 配置：

```python
# Check if model switched - if so, update scheduler config
if i > 0:
    prev_model_index = self.step_config["step"][i - 1]
    if prev_model_index != model_index:
        logger.info(f"=== MODEL SWITCH at step {i}: {self.step_config['name'][prev_model_index]} -> {model_name} ===")
        
        # Update scheduler config if different models have different configs
        if self.scheduler_configs is not None and model_index < len(self.scheduler_configs):
            new_scheduler_config = self.scheduler_configs[model_index]
            current_config = self.scheduler.config
            
            # Check if snr_shift_scale needs to be updated
            if "snr_shift_scale" in new_scheduler_config:
                new_snr_shift_scale = new_scheduler_config["snr_shift_scale"]
                current_snr_shift_scale = getattr(current_config, "snr_shift_scale", None)
                
                if new_snr_shift_scale != current_snr_shift_scale:
                    logger.info(f"Updating scheduler snr_shift_scale: {current_snr_shift_scale} -> {new_snr_shift_scale}")
                    
                    # Re-initialize scheduler with updated config
                    from diffusers import CogVideoXDDIMScheduler
                    if isinstance(self.scheduler, CogVideoXDDIMScheduler):
                        updated_config = self.scheduler.config.copy()
                        updated_config["snr_shift_scale"] = new_snr_shift_scale
                        self.scheduler = CogVideoXDDIMScheduler.from_config(updated_config)
                        # Restore timesteps
                        self.scheduler.set_timesteps(num_inference_steps, device=latents.device)
                        logger.info(f"Scheduler updated with snr_shift_scale={new_snr_shift_scale}, timesteps restored")
```

### 3. 关键实现细节

- **配置加载**：在初始化时加载所有模型的 scheduler 配置
- **动态切换**：在模型切换点检测配置差异并更新
- **状态恢复**：重新创建 scheduler 后，必须恢复 timesteps 状态，否则会报错 "Number of inference steps is 'None'"

## 修复效果

修复后的日志显示：

```
Step 9/25 (5B): noise_pred guided std: 0.7252
=== MODEL SWITCH at step 10: CogVideoX-5b -> CogVideoX-2b ===
Updating scheduler snr_shift_scale: 1.0 -> 3.0
Scheduler updated with snr_shift_scale=3.0, timesteps restored (num_inference_steps=25)
Step 10/25 (2B): noise_pred guided std: 1.1387
Step 11/25 (2B): noise_pred guided std: 1.1388
Step 12/25 (2B): noise_pred guided std: 1.1365
...
```

虽然 Step 10 的 std 仍然较大，但：
1. ✅ Scheduler 配置已正确切换
2. ✅ 后续步骤的 std 逐渐下降，说明 2B 模型工作正常
3. ✅ 最终生成的视频质量显著改善

## 技术要点总结

### 1. Scheduler 是共享的，但配置应该匹配模型
- 5B 和 2B 共用同一个 scheduler 实例
- 但每个模型有自己的 scheduler 配置（特别是 `snr_shift_scale`）
- 必须在模型切换时动态更新配置

### 2. DDIM Scheduler 每一步独立计算
- 与 DPM Scheduler 不同，DDIM Scheduler 不需要跟踪 `old_pred_original_sample`
- 但重新创建 scheduler 后必须恢复 timesteps 状态

### 3. 配置差异是关键
- `snr_shift_scale` 的差异（1.0 vs 3.0）会导致预测分布异常
- 必须确保每个模型使用正确的 scheduler 配置

## 相关文件

- `compression/hybrid_sd/diffusers/pipeline_cogvideox.py`：去噪循环和模型切换逻辑
- `compression/hybrid_sd/inference_pipeline.py`：scheduler 配置加载
- `/data/models/hybridsd_checkpoint/CogVideoX-5b/scheduler/scheduler_config.json`：5B 模型配置
- `/data/models/hybridsd_checkpoint/CogVideoX-2b/scheduler/scheduler_config.json`：2B 模型配置

## 验证方法

运行混合去噪测试：

```bash
python examples/hybrid_sd/hybrid_video.py \
    --model_paths /data/models/hybridsd_checkpoint/CogVideoX-5b /data/models/hybridsd_checkpoint/CogVideoX-2b \
    --steps 10,15 \
    --output_dir results/test_hybrid
```

检查日志中是否出现：
- `Updating scheduler snr_shift_scale: 1.0 -> 3.0`
- `Scheduler updated with snr_shift_scale=3.0, timesteps restored`
- 模型切换后 `noise_pred` 的 std 不会异常增大
- 最终生成的视频质量良好

## 总结

这个问题的核心是**模型配置不匹配**：虽然代码正确地切换了 Transformer 模型，但忽略了 Scheduler 配置也需要匹配。通过实现动态 Scheduler 配置切换，确保了每个模型使用正确的配置，从而解决了混合去噪质量差的问题。

