# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import gc
import json
import os 
import time
from peft import LoraModel, LoraConfig, set_peft_model_state_dict
from typing import Union, List, Optional
from PIL import Image
from packaging import version
import diffusers
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel

from diffusers.utils.import_utils import is_xformers_available
from .diffusers.pipeline_stable_diffusion import StableDiffusionPipeline, HybridStableDiffusionPipeline
from .diffusers.pipeline_stable_diffusion_xl import  HybridStableDiffusionXLPipeline
from .diffusers.pipeline_cogvideox import HybridCogVideoXPipeline
# Optional import for LCM pipeline (only needed for hybrid_LCM.py)
try:
    from .diffusers.pipline_hybrid_LCM import HybridLCMPipeline
except ImportError:
    HybridLCMPipeline = None
from compression.prune_sd.models.unet_2d_condition import UNet2DConditionModel as CustomUNet2DConditionModel
try:
    from compression.prune_sd.LCM.Scheduling_LCM import LCMScheduler
except ImportError:
    LCMScheduler = None
diffusers_version = int(diffusers.__version__.split('.')[1])


class TimingProfiler:
    """用于测量模型各组件执行时间的Profiler"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """重置所有计时数据"""
        from collections import defaultdict
        self._layer_times = defaultdict(float)
        self._hooks = []
        self._current_step = 0
        self._step_layer_times_start = {}  # 每步开始时的layer times
        self._step_times = []
        
    def install_hooks(self, transformer):
        """安装计时hooks到transformer的每一层"""
        self.remove_hooks()
        import time as time_module
        self._time_module = time_module
        
        # 为每个layer安装hooks
        for layer_idx, block in enumerate(transformer.blocks):
            # Self-attention (attn1)
            if hasattr(block, 'attn1') and block.attn1 is not None:
                original_attn1 = block.attn1.forward
                def make_self_attn_hook(layer_idx, orig_fn):
                    def hooked(*args, **kwargs):
                        t0 = self._time_module.perf_counter()
                        result = orig_fn(*args, **kwargs)
                        t1 = self._time_module.perf_counter()
                        self._layer_times[f'self_attn_layer_{layer_idx}'] += (t1 - t0)
                        return result
                    return hooked
                block.attn1.forward = make_self_attn_hook(layer_idx, original_attn1)
                self._hooks.append(('attn1', layer_idx, original_attn1, block.attn1))
            
            # Cross-attention (attn2)
            if hasattr(block, 'attn2') and block.attn2 is not None:
                original_attn2 = block.attn2.forward
                def make_cross_attn_hook(layer_idx, orig_fn):
                    def hooked(*args, **kwargs):
                        t0 = self._time_module.perf_counter()
                        result = orig_fn(*args, **kwargs)
                        t1 = self._time_module.perf_counter()
                        self._layer_times[f'cross_attn_layer_{layer_idx}'] += (t1 - t0)
                        return result
                    return hooked
                block.attn2.forward = make_cross_attn_hook(layer_idx, original_attn2)
                self._hooks.append(('attn2', layer_idx, original_attn2, block.attn2))
            
            # FFN
            if hasattr(block, 'ffn') and block.ffn is not None:
                original_ffn = block.ffn.forward
                def make_ffn_hook(layer_idx, orig_fn):
                    def hooked(*args, **kwargs):
                        t0 = self._time_module.perf_counter()
                        result = orig_fn(*args, **kwargs)
                        t1 = self._time_module.perf_counter()
                        self._layer_times[f'ffn_layer_{layer_idx}'] += (t1 - t0)
                        return result
                    return hooked
                block.ffn.forward = make_ffn_hook(layer_idx, original_ffn)
                self._hooks.append(('ffn', layer_idx, original_ffn, block.ffn))
                
        print(f"[TimingProfiler] 已安装 {len(self._hooks)} 个hooks")
        
    def remove_hooks(self):
        """移除所有hooks"""
        for attr_name, layer_idx, original_fn, block in self._hooks:
            try:
                if attr_name == 'attn1':
                    block.forward = original_fn
                elif attr_name == 'attn2':
                    block.forward = original_fn
                elif attr_name == 'ffn':
                    block.forward = original_fn
            except:
                pass
        self._hooks = []
        
    def start_step(self):
        """记录每步开始时的layer times"""
        # 创建当前layer times的副本
        self._step_layer_times_start = dict(self._layer_times)
        
    def get_step_component_times(self):
        """获取当前step各组件的时间（从上次start_step后的增量）"""
        total_self_attn = 0
        total_cross_attn = 0
        total_ffn = 0
        
        for key, current_val in self._layer_times.items():
            # 获取该key在上一步开始时的值
            start_val = self._step_layer_times_start.get(key, 0)
            delta = current_val - start_val
            
            if 'self_attn' in key:
                total_self_attn += delta
            elif 'cross_attn' in key:
                total_cross_attn += delta
            elif 'ffn' in key:
                total_ffn += delta
        
        return {
            'self_attn': total_self_attn,
            'cross_attn': total_cross_attn,
            'ffn': total_ffn,
            'transformer': total_self_attn + total_cross_attn + total_ffn
        }
    
    def get_total_component_times(self):
        """获取各组件的总时间（所有step累计）"""
        total_self_attn = sum(v for k, v in self._layer_times.items() if 'self_attn' in k)
        total_cross_attn = sum(v for k, v in self._layer_times.items() if 'cross_attn' in k)
        total_ffn = sum(v for k, v in self._layer_times.items() if 'ffn' in k)
        
        return {
            'self_attn': total_self_attn,
            'cross_attn': total_cross_attn,
            'ffn': total_ffn,
            'transformer': total_self_attn + total_cross_attn + total_ffn
        }
    
    def print_step_timing(self, step_idx, num_steps, model_name, latent_shape, step_time):
        """打印每步的详细计时信息"""
        # 计算各组件时间（当前step的增量）
        component_times = self.get_step_component_times()
        total_self_attn = component_times['self_attn']
        total_cross_attn = component_times['cross_attn']
        total_ffn = component_times['ffn']
        transformer_time = component_times['transformer']
        
        # 计算其他时间 (step_time - transformer_time)
        other_time = step_time - transformer_time
        if other_time < 0:
            other_time = step_time * 0.3  # 估算
        
        # 计算序列长度
        seq_length = latent_shape[1] * latent_shape[3] * latent_shape[4]
        
        print(f"\n{'='*80}")
        print(f"Step {step_idx+1}/{num_steps} - {model_name}")
        print(f"{'='*80}")
        print(f"输入数据维度: {tuple(latent_shape)}")
        print(f"  - Batch: {latent_shape[0]}, Time: {latent_shape[1]}, Channel: {latent_shape[2]}")
        print(f"  - Height: {latent_shape[3]}, Width: {latent_shape[4]}")
        print(f"  - 序列长度: {seq_length}")
        print(f"\n时间组成:")
        print(f"  Self Attention:  {total_self_attn:.3f}s ({total_self_attn/step_time*100:.1f}%)")
        print(f"  Cross Attention: {total_cross_attn:.3f}s ({total_cross_attn/step_time*100:.1f}%)")
        print(f"  FFN:            {total_ffn:.3f}s ({total_ffn/step_time*100:.1f}%)")
        print(f"  Transformer总计: {transformer_time:.3f}s ({transformer_time/step_time*100:.1f}%)")
        print(f"  Scheduler+其他: {other_time:.3f}s ({other_time/step_time*100:.1f}%)")
        print(f"  ──────────────────────────────────")
        print(f"  Step总时间:     {step_time:.3f}s")
        print(f"{'='*80}\n")
        
        # 存储step时间
        self._step_times.append({
            'step': step_idx + 1,
            'total_time': step_time,
            'self_attn': total_self_attn,
            'cross_attn': total_cross_attn,
            'ffn': total_ffn,
            'transformer': transformer_time,
            'other': other_time
        })
        
    def get_total_step_time(self):
        """获取所有step的总时间"""
        return sum(s['total_time'] for s in self._step_times)

class InferencePipeline:
    def __init__(self, weight_folder, seed, device, args):
        self.weight_folder = weight_folder
        self.seed = seed
        self.device = torch.device(device)
        self.args = args

        self.pipe = None
        self.generator = None

    def clear(self) -> None:
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()

    def set_pipe_and_generator(self): 
        # disable NSFW filter to avoid black images, **ONLY for the benchmark evaluation** 
        if diffusers_version == 15: # for the specified version in requirements.txt
            self.pipe = StableDiffusionPipeline.from_pretrained(self.weight_folder,
                                                                torch_dtype=torch.float16).to(self.device)
            self.pipe.safety_checker = lambda images, clip_input: (images, False) 
        elif diffusers_version >= 19: # for recent diffusers versions
            self.pipe = StableDiffusionPipeline.from_pretrained(self.weight_folder,
                                                                safety_checker=None, torch_dtype=torch.float16).to(self.device)
        else: # for the versions between 0.15 and 0.19, the benchmark scores are not guaranteed.
            raise Exception(f"Use diffusers version as either ==0.15.0 or >=0.19 (from current {diffusers.__version__})")

        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers
                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    self.args.logger.log(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.pipe.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if self.args.use_dpm_solver:    
            self.args.logger.log(" ** replace PNDM scheduler into DPM-Solver")
            from diffusers import DPMSolverMultistepScheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)        

    def set_lora_ckpt(self): 
        if self.args.is_lora_checkpoint:
            self.args.logger.log(" ** use lora checkpoints")
            load_and_set_lora_ckpt(
                               pipe=self.pipe,
                               weight_path=os.path.join(self.args.lora_weight_path, 'lora.pt'),
                               config_path=os.path.join(self.args.lora_weight_path, 'lora_config.json'),
                               dtype=torch.float16)

    def generate(self, prompt: Union[str, List[str]],  negative_prompt: Union[str, List[str]] = None, n_steps: int = 25, img_sz: int = 512,  guidance_scale: float = 7.5, num_images_per_prompt: int = 1, save_path: str = None) -> List[Image.Image]:
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            height = img_sz,
            width = img_sz,
            generator=self.generator,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            save_path=save_path
        )
        return out.images
    
    def _count_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def get_sdm_params(self):
        params_unet = self._count_params(self.pipe.unet)
        params_text_enc = self._count_params(self.pipe.text_encoder)
        params_image_dec = self._count_params(self.pipe.vae.decoder)
        params_total = params_unet + params_text_enc + params_image_dec 
        return f"Total {(params_total/1e6):.1f}M (U-Net {(params_unet/1e6):.1f}M; TextEnc {(params_text_enc/1e6):.1f}M; ImageDec {(params_image_dec/1e6):.1f}M)"


class HybridInferencePipeline:
    def __init__(self, weight_folders, seed, device, args):
        self.weight_folders = weight_folders
        self.device = torch.device(device)
        self.seed = seed
        self.args = args
        self.args.vae_path = None
        self.pipe = None
        self.generator = None

    def clear(self) -> None:
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()

    def set_pipe_and_generator(self): 
        # disable NSFW filter to avoid black images, **ONLY for the benchmark evaluation** 
        text_encoder = CLIPTextModel.from_pretrained(
                self.weight_folders[0], subfolder="text_encoder"
            ).to(self.device, dtype=torch.float16).requires_grad_(False)
        if self.args.vae_path is not None:
            from diffusers import AutoencoderTiny
            print("loading Tiny VAE")
            vae = AutoencoderTiny.from_pretrained(self.args.vae_path).to(self.device, dtype=torch.float16).requires_grad_(False)
        else:
            if 'Realistic_Vision' in self.weight_folders[0]:
                print("loading stabilityai/sd-vae-ft-ema...")
                vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device, dtype=torch.float16).requires_grad_(False)
            else:
                vae = AutoencoderKL.from_pretrained(
                        self.weight_folders[0], subfolder="vae"
                    ).to(self.device, dtype=torch.float16).requires_grad_(False)
        tokenizer = CLIPTokenizer.from_pretrained(
                    self.weight_folders[0], subfolder="tokenizer"
            )
        unets = []
        for path in self.weight_folders:
            if 'hybrid-sd' in path:
                MODEL_OBJ = CustomUNet2DConditionModel
            else:
                MODEL_OBJ = UNet2DConditionModel

            unets.append(
                MODEL_OBJ.from_pretrained(
                    path, subfolder="unet"
                ).to(self.device, dtype=torch.float16).requires_grad_(False)
            )

        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers
                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    self.args.logger.log(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                for unet in unets:
                    unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        self.pipe = HybridStableDiffusionPipeline.from_pretrained(
            self.weight_folders[0],
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unets[0]
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.unets = unets
        total_step, step_config = self.get_step_config(self.args)
        print(f'total_step={total_step}, step_config={step_config}')
        self.pipe.step_config = step_config
        self.total_step = total_step 
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
    
        if self.args.use_dpm_solver:    
            # self.args.logger.info(" ** Use DPMSolverMultistepScheduler")
            from diffusers import DPMSolverMultistepScheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)        

        if self.args.use_pndm_solver:    
            # self.args.logger.info(" ** Use PNDMScheduler")
            from diffusers import PNDMScheduler
            self.pipe.scheduler = PNDMScheduler.from_config(self.pipe.scheduler.config)        
        
        # skip safety_checker
        self.pipe.safety_checker = None

    def set_lora_ckpt(self):
        if self.args.is_lora_checkpoint:
            self.args.logger.log(" ** use lora checkpoints")
            hybrid_load_and_set_lora_ckpt(
                                pipe=self.pipe,
                                weight_path=[os.path.join(path, 'lora.pt') for path in self.args.lora_weight_path],
                                config_path=[os.path.join(path, 'lora_config.json') for path in self.args.lora_weight_path],
                                dtype=torch.float16)


    def generate(self, prompt: Union[str, List[str]],  negative_prompt: Union[str, List[str]] = None, img_sz: int = 512,  guidance_scale: float = 7.5, num_images_per_prompt=1, save_path=None,prompt_embeds=None, negative_prompt_embeds=None) -> List[Image.Image]:
        out = self.pipe(
            prompt = prompt,
            negative_prompt = negative_prompt,
            prompt_embeds = prompt_embeds,
            negative_prompt_embeds = negative_prompt_embeds,
            num_inference_steps = self.total_step,
            height = img_sz,
            width = img_sz,
            generator = self.generator,
            guidance_scale = guidance_scale,
            num_images_per_prompt = num_images_per_prompt,
            save_path = save_path,
        )
        return out.images
    
    def _count_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def get_step_config(self, args):
        assert len(args.steps) > 0 
        total_step = sum(args.steps)
        assert total_step > 0
        assert len(self.weight_folders) == len(args.steps)
        step_config = {
            "step":{},
            "name":{}
        }
        total_step = 0 
        for index, model_step in enumerate(args.steps):
            for i in range(model_step):
                step_config["step"][total_step] = index
                total_step += 1
        for index, model_name in enumerate(self.weight_folders):
            step_config["name"][index] = model_name.split("/")[-1]

        return total_step, step_config

    def get_sdm_params(self):
        params_str = ""
        for index in range(len(self.pipe.unets)):
            model_name = self.weight_folders[index].split("/")[-1]
            cur_unet = self._count_params(self.pipe.unets[index])
            params_str += f" {model_name}: {(cur_unet/1e6):.1f}M"
        params_text_enc = self._count_params(self.pipe.text_encoder)
        params_image_dec = self._count_params(self.pipe.vae.decoder)
        return_str =  params_str + f"TextEnc {(params_text_enc/1e6):.1f}M; ImageDec {(params_image_dec/1e6):.1f}M)"
        return return_str


class HybridSDXLInferencePipeline:
    def __init__(self, weight_folders, seed, device, args):
        self.weight_folders = weight_folders
        self.device = torch.device(device)
        self.seed = seed
        self.args = args
        self.args.vae_path = None
        self.pipe = None
        self.generator = None
        if args.weight_dtype == "fp16":
            self.weight_dtype = torch.float32 
        elif args.weight_dtype == "fp16":
            self.weight_dtype = torch.float16

    def clear(self) -> None:
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()

    def set_pipe_and_generator(self): 
        # disable NSFW filter to avoid black images, **ONLY for the benchmark evaluation**         
        if self.args.vae_path is not None:
            from diffusers import AutoencoderTiny
            print("loading Tiny VAE")
            vae = AutoencoderTiny.from_pretrained(self.args.vae_path).to(self.device, dtype=torch.float16).requires_grad_(False)
        else:
            vae = AutoencoderKL.from_pretrained(
                    "/data/models/hybridsd_checkpoint/madebyollin--sdxl-vae-fp16-fix", subfolder="vae"
                ).to(self.device, dtype=torch.float16).requires_grad_(False)
        unets = []
        for path in self.weight_folders:
            MODEL_OBJ = UNet2DConditionModel
            unets.append(
                MODEL_OBJ.from_pretrained(
                    path, subfolder="unet"
                ).to(self.device, dtype=torch.float16).requires_grad_(False)
            )

        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers
                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    self.args.logger.log(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                for unet in unets:
                    unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        self.pipe = HybridStableDiffusionXLPipeline.from_pretrained(
            self.weight_folders[0],
            vae=vae,
            unet=unets[0],
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.unets = unets
        total_step, step_config = self.get_step_config(self.args)
        print(f'total_step={total_step}, step_config={step_config}')
        self.pipe.step_config = step_config
        self.total_step = total_step 
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.pipe.to(self.device)
    
        if self.args.use_dpm_solver:    
            # self.args.logger.info(" ** Use DPMSolverMultistepScheduler")
            from diffusers import DPMSolverMultistepScheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)        

        if self.args.use_pndm_solver:    
            # self.args.logger.info(" ** Use PNDMScheduler")
            from diffusers import PNDMScheduler
            self.pipe.scheduler = PNDMScheduler.from_config(self.pipe.scheduler.config)        
        
        # skip safety_checker
        self.pipe.safety_checker = None

    def set_lora_ckpt(self):
        if self.args.is_lora_checkpoint:
            self.args.logger.log(" ** use lora checkpoints")
            hybrid_load_and_set_lora_ckpt(
                                pipe=self.pipe,
                                weight_path=[os.path.join(path, 'lora.pt') for path in self.args.lora_weight_path],
                                config_path=[os.path.join(path, 'lora_config.json') for path in self.args.lora_weight_path],
                                dtype=torch.float16)


    def generate(self, prompt: Union[str, List[str]],  negative_prompt: Union[str, List[str]] = None, img_sz: int = 512,  guidance_scale: float = 7.5, num_images_per_prompt=1, save_path=None, prompt_embeds=None, negative_prompt_embeds=None, image_prompt_embeds=None, uncond_image_prompt_embeds=None) -> List[Image.Image]:
        out = self.pipe(
            prompt = prompt,
            negative_prompt = negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps = self.total_step,
            height = img_sz,
            width = img_sz,
            generator = self.generator,
            guidance_scale = guidance_scale,
            num_images_per_prompt = num_images_per_prompt,
            save_path = save_path,
            image_prompt_embeds=image_prompt_embeds,
            uncond_image_prompt_embeds=uncond_image_prompt_embeds,
        )
        return out.images
    
    def _count_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def get_step_config(self, args):
        assert len(args.steps) > 0 
        total_step = sum(args.steps)
        assert total_step > 0
        assert len(self.weight_folders) == len(args.steps)
        step_config = {
            "step":{},
            "name":{}
        }
        total_step = 0 
        for index, model_step in enumerate(args.steps):
            for i in range(model_step):
                step_config["step"][total_step] = index
                total_step += 1
        for index, model_name in enumerate(self.weight_folders):
            step_config["name"][index] = model_name.split("/")[-1]

        return total_step, step_config

    def get_sdm_params(self):
        params_str = ""
        for index in range(len(self.pipe.unets)):
            model_name = self.weight_folders[index].split("/")[-1]
            cur_unet = self._count_params(self.pipe.unets[index])
            params_str += f" {model_name}: {(cur_unet/1e6):.1f}M"
        params_text_enc = self._count_params(self.pipe.text_encoder)
        params_image_dec = self._count_params(self.pipe.vae.decoder)
        return_str =  params_str + f"TextEnc {(params_text_enc/1e6):.1f}M; ImageDec {(params_image_dec/1e6):.1f}M)"
        return return_str




class HybridLCMInferencePipeline:
    def __init__(self, weight_folders, seed, device, args):
        if HybridLCMPipeline is None:
            raise ImportError("HybridLCMPipeline is not available. This pipeline is only for LCM models.")
        if LCMScheduler is None:
            raise ImportError("LCMScheduler is not available. Please install the required LCM dependencies.")
        self.weight_folders = weight_folders
        self.device = torch.device(device)
        self.seed = seed
        self.args = args
        self.args.vae_path = None
        self.pipe = None
        self.generator = None
        self.scheduler = LCMScheduler.from_pretrained(args.pretrained_teacher_model, subfolder="scheduler")
        self.pretrained_teacher_model = args.pretrained_teacher_model


    def clear(self) -> None:
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()

    def set_pipe_and_generator(self): 
        # disable NSFW filter to avoid black images, **ONLY for the benchmark evaluation** 
        text_encoder = CLIPTextModel.from_pretrained(
                self.pretrained_teacher_model, subfolder="text_encoder"
            ).to(self.device, dtype=torch.float16).requires_grad_(False)
        if self.args.vae_path is not None:
            from diffusers import AutoencoderTiny
            print("loading Tiny VAE")
            vae = AutoencoderTiny.from_pretrained(self.args.vae_path).to(self.device, dtype=torch.float16).requires_grad_(False)
        else:
            if 'Realistic_Vision' in self.pretrained_teacher_model:
                print("loading stabilityai/sd-vae-ft-ema...")
                vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device, dtype=torch.float16).requires_grad_(False)
            else:
                vae = AutoencoderKL.from_pretrained(
                        self.pretrained_teacher_model, subfolder="vae"
                    ).to(self.device, dtype=torch.float16).requires_grad_(False)
        tokenizer = CLIPTokenizer.from_pretrained(
                    self.pretrained_teacher_model, subfolder="tokenizer"
            )
        unets = []
        for path in self.weight_folders:
            if 'prune' in path or 'Prune' in path or 'ours' in path:
                MODEL_OBJ = CustomUNet2DConditionModel
            else:
                MODEL_OBJ = UNet2DConditionModel

            unets.append(
                MODEL_OBJ.from_pretrained(
                    path, subfolder="unet"
                ).to(self.device, dtype=torch.float16).requires_grad_(False)
            )

        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers
                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    self.args.logger.log(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                for unet in unets:
                    unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if HybridLCMPipeline is None:
            raise ImportError("HybridLCMPipeline is not available. This pipeline requires the LCM pipeline module.")
        self.pipe = HybridLCMPipeline.from_pretrained(
            self.pretrained_teacher_model,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=self.scheduler, # using LCM scheduler
            unet=unets[0]
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.unets = unets
        total_step, step_config = self.get_step_config(self.args)
        print(f'total_step={total_step}, step_config={step_config}')
        self.pipe.step_config = step_config
        self.total_step = total_step 
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
    
        
        # skip safety_checker
        self.pipe.safety_checker = None

    def set_lora_ckpt(self):
        if self.args.is_lora_checkpoint:
            self.args.logger.log(" ** use lora checkpoints")
            hybrid_load_and_set_lora_ckpt(
                                pipe=self.pipe,
                                weight_path=[os.path.join(path, 'lora.pt') for path in self.args.lora_weight_path],
                                config_path=[os.path.join(path, 'lora_config.json') for path in self.args.lora_weight_path],
                                dtype=torch.float16)


    def generate(self, prompt: Union[str, List[str]],  negative_prompt: Union[str, List[str]] = None, img_sz: int = 512,  guidance_scale: float = 7.5, num_images_per_prompt=1, save_path=None) -> List[Image.Image]:
        out = self.pipe(
            prompt = prompt,
            negative_prompt = negative_prompt,
            num_inference_steps = self.total_step,
            height = img_sz,
            width = img_sz,
            generator = self.generator,
            guidance_scale = guidance_scale,
            num_images_per_prompt = num_images_per_prompt,
            save_path = save_path
        )
        return out.images
    
    def generate_latents(self, prompt: Union[str, List[str]],  negative_prompt: Union[str, List[str]] = None, img_sz: int = 512,  guidance_scale: float = 7.5, num_images_per_prompt=1, output_type = "latent", save_path=None) -> List[Image.Image]:
        out = self.pipe(
            prompt = prompt,
            negative_prompt = negative_prompt,
            num_inference_steps = self.total_step,
            height = img_sz,
            width = img_sz,
            output_type = "latent",
            generator = self.generator,
            guidance_scale = guidance_scale,
            num_images_per_prompt = num_images_per_prompt,
            save_path = save_path
        )
        return out.images
    
    def _count_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def get_step_config(self, args):
        assert len(args.steps) > 0 
        total_step = sum(args.steps)
        assert total_step > 0
        assert len(self.weight_folders) == len(args.steps)
        step_config = {
            "step":{},
            "name":{}
        }
        total_step = 0 
        for index, model_step in enumerate(args.steps):
            for i in range(model_step):
                step_config["step"][total_step] = index
                total_step += 1
        for index, model_name in enumerate(self.weight_folders):
            step_config["name"][index] = model_name.split("/")[-1]

        return total_step, step_config

    def get_sdm_params(self):
        params_str = ""
        for index in range(len(self.pipe.unets)):
            model_name = self.weight_folders[index].split("/")[-1]
            cur_unet = self._count_params(self.pipe.unets[index])
            params_str += f" {model_name}: {(cur_unet/1e6):.1f}M"
        params_text_enc = self._count_params(self.pipe.text_encoder)
        params_image_dec = self._count_params(self.pipe.vae.decoder)
        return_str =  params_str + f"TextEnc {(params_text_enc/1e6):.1f}M; ImageDec {(params_image_dec/1e6):.1f}M)"
        return return_str




def load_and_set_lora_ckpt(pipe, weight_path, config_path, dtype):
    device = pipe.unet.device

    with open(config_path, "r") as f:
        lora_config = json.load(f)
    lora_checkpoint_sd = torch.load(weight_path, map_location=device)
    unet_lora_ds = {k: v for k, v in lora_checkpoint_sd.items() if "text_encoder_" not in k}
    text_encoder_lora_ds = {
        k.replace("text_encoder_", ""): v for k, v in lora_checkpoint_sd.items() if "text_encoder_" in k
    }

    unet_config = LoraConfig(**lora_config["peft_config"])
    pipe.unet = LoraModel(unet_config, pipe.unet)
    set_peft_model_state_dict(pipe.unet, unet_lora_ds)

    if "text_encoder_peft_config" in lora_config:
        text_encoder_config = LoraConfig(**lora_config["text_encoder_peft_config"])
        pipe.text_encoder = LoraModel(text_encoder_config, pipe.text_encoder)
        set_peft_model_state_dict(pipe.text_encoder, text_encoder_lora_ds)

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()

    pipe.to(device)
    return pipe


def hybrid_load_and_set_lora_ckpt(pipe, weight_paths, config_paths, dtype):
    device = pipe.unet.device

    assert len(config_paths) == len(weight_paths)
    assert len(config_paths) == pipe.unets
    
    for index in len(config_paths):
        config_path = config_paths[index]
        weight_path = weight_paths[index]
        with open(config_path, "r") as f:
            lora_config = json.load(f)
        lora_checkpoint_sd = torch.load(weight_path, map_location=device)
        unet_lora_ds = {k: v for k, v in lora_checkpoint_sd.items() if "text_encoder_" not in k}
        text_encoder_lora_ds = {
            k.replace("text_encoder_", ""): v for k, v in lora_checkpoint_sd.items() if "text_encoder_" in k
        }

        unet_config = LoraConfig(**lora_config["peft_config"])
        pipe.unets[index] = LoraModel(unet_config, pipe.unets[index])
        set_peft_model_state_dict(pipe.unets[index], unet_lora_ds)

        if "text_encoder_peft_config" in lora_config:
            text_encoder_config = LoraConfig(**lora_config["text_encoder_peft_config"])
            pipe.text_encoder = LoraModel(text_encoder_config, pipe.text_encoder)
            set_peft_model_state_dict(pipe.text_encoder, text_encoder_lora_ds)

        if dtype in (torch.float16, torch.bfloat16):
            pipe.unets[index].half()
            pipe.text_encoder.half()

        pipe.to(device)
        return pipe


class HybridVideoInferencePipeline:
    """
    Pipeline for text-to-video generation using CogVideoX with collaborative inference.
    Supports switching between multiple transformers (e.g., CogVideoX-5B and CogVideoX-2B) during denoising.
    """
    
    def __init__(self, weight_folders, seed, device, args):
        self.weight_folders = weight_folders
        self.device = torch.device(device)
        # Support separate VAE device for multi-GPU setup
        # If vae_device is not specified, use the same device as main device
        if hasattr(args, 'vae_device') and args.vae_device:
            self.vae_device = torch.device(args.vae_device)
        else:
            # Auto-select: use next available GPU (e.g., if device is cuda:0, use cuda:1)
            if self.device.type == 'cuda' and torch.cuda.device_count() > 1:
                current_gpu = self.device.index if self.device.index is not None else 0
                next_gpu = (current_gpu + 1) % torch.cuda.device_count()
                self.vae_device = torch.device(f'cuda:{next_gpu}')
                print(f"Auto-selected VAE device: {self.vae_device} (main device: {self.device})")
            else:
                self.vae_device = self.device
        self.seed = seed
        self.args = args
        self.pipe = None
        self.generator = None
        
        # 初始化 TimingProfiler
        self.timing_profiler = TimingProfiler()
        self._profiling_enabled = getattr(args, 'enable_profiling', False)
    
    def clear(self) -> None:
        """Clear pipeline and free memory"""
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()
    
    def set_pipe_and_generator(self):
        """
        Load models and create HybridCogVideoXPipeline.
        Loads shared components (Text Encoder, VAE) and multiple transformers.
        """
        # Import CogVideoX components
        from transformers import T5EncoderModel, T5Tokenizer
        from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel
        from diffusers import CogVideoXPipeline
        
        # 1. Load shared Text Encoder (T5)
        print(f"Loading Text Encoder from {self.weight_folders[0]}")
        text_encoder = T5EncoderModel.from_pretrained(
            self.weight_folders[0], subfolder="text_encoder"
        ).to(self.device, dtype=torch.float16).requires_grad_(False)
        
        tokenizer = T5Tokenizer.from_pretrained(
            self.weight_folders[0], subfolder="tokenizer"
        )
        
        # 2. Load shared VAE on separate GPU if available
        print(f"Loading VAE from {self.weight_folders[0]} on device {self.vae_device}")
        vae = AutoencoderKLCogVideoX.from_pretrained(
            self.weight_folders[0], subfolder="vae"
        ).to(self.vae_device, dtype=torch.float16).requires_grad_(False)
        # Enable VAE memory optimizations to reduce memory usage
        vae.enable_slicing()
        vae.enable_tiling()
        print(f"VAE slicing and tiling enabled for memory optimization (VAE on {self.vae_device})")
        
        # 3. Load multiple transformers
        print(f"Loading {len(self.weight_folders)} transformers...")
        transformers = []
        for path in self.weight_folders:
            print(f"  Loading transformer from {path}")
            transformer = CogVideoXTransformer3DModel.from_pretrained(
                path, subfolder="transformer", torch_dtype=torch.float16
            ).to(self.device).requires_grad_(False)
            # Ensure all modules are in float16
            transformer = transformer.half()
            transformers.append(transformer)
        
            # 4. Enable xformers if requested
            # NOTE: xformers may cause issues with CogVideoX transformer, so we disable it for now
            # The error "The size of tensor a (226) must match the size of tensor b (17550)" 
            # occurs even with the parent class when xformers is enabled
            if self.args.enable_xformers_memory_efficient_attention:
                print("Warning: xformers is requested but disabled for CogVideoX due to compatibility issues")
                print("The pipeline will run without xformers memory efficient attention")
                # Temporarily disable xformers for CogVideoX
                # if is_xformers_available():
                #     import xformers
                #     xformers_version = version.parse(xformers.__version__)
                #     if xformers_version == version.parse("0.0.16"):
                #         if hasattr(self.args, 'logger') and self.args.logger:
                #             self.args.logger.log(
                #                 "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                #             )
                #     for transformer in transformers:
                #         transformer.enable_xformers_memory_efficient_attention()
                # else:
                #     raise ValueError("xformers is not available. Make sure it is installed correctly")
        
        # 5. Create base pipeline first, then convert to Hybrid
        print("Creating HybridCogVideoXPipeline...")
        # Create base pipeline with the first transformer to ensure proper initialization
        base_pipeline = CogVideoXPipeline.from_pretrained(
            self.weight_folders[0],
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
            transformer=transformers[0],
            torch_dtype=torch.float16
        )
        
        # Convert to HybridCogVideoXPipeline
        # Use the transformer from base_pipeline to ensure it's properly configured
        self.pipe = HybridCogVideoXPipeline(
            transformer=base_pipeline.transformer,  # Use pipeline's transformer (properly configured)
            vae=base_pipeline.vae,
            text_encoder=base_pipeline.text_encoder,
            tokenizer=base_pipeline.tokenizer,
            scheduler=base_pipeline.scheduler
        )
        
        # 6. Set multiple transformers
        # Replace the first transformer with the one from pipeline (if needed)
        # and add additional transformers
        # Ensure all transformers are properly loaded with the same configuration
        pipeline_transformers = [base_pipeline.transformer]  # First transformer from pipeline
        for i in range(1, len(transformers)):
            # Load additional transformers with the same method as pipeline
            additional_transformer = CogVideoXTransformer3DModel.from_pretrained(
                self.weight_folders[i], 
                subfolder="transformer",
                torch_dtype=torch.float16
            ).to(self.device).requires_grad_(False)
            pipeline_transformers.append(additional_transformer)
        
        self.pipe.set_transformers(pipeline_transformers)
        
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
            else:
                # Fallback: use current scheduler config
                scheduler_configs.append(self.pipe.scheduler.config)
                print(f"Warning: scheduler config not found at {scheduler_config_path}, using current scheduler config")
        
        self.pipe.set_scheduler_configs(scheduler_configs)
        
        # 8. Set step configuration
        total_step, step_config = self.get_step_config(self.args)
        print(f'total_step={total_step}, step_config={step_config}')
        self.pipe.set_step_config(step_config)
        self.total_step = total_step
        
        # 9. Set generator
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        
        # 10. Configure scheduler if needed
        if hasattr(self.args, 'use_dpm_solver') and self.args.use_dpm_solver:
            from diffusers import CogVideoXDPMScheduler
            self.pipe.scheduler = CogVideoXDPMScheduler.from_config(
                self.pipe.scheduler.config, timestep_spacing="trailing"
            )
        
        # 11. Disable progress bar if needed
        if hasattr(self.pipe, 'set_progress_bar_config'):
            self.pipe.set_progress_bar_config(disable=True)
        
        # 12. 安装 timing hooks（如果启用）
        if self._profiling_enabled:
            print("启用性能分析...")
            self.timing_profiler.reset()
            # 为第一个transformer安装hooks
            if self.pipe.transformers and len(self.pipe.transformers) > 0:
                self.timing_profiler.install_hooks(self.pipe.transformers[0])
                print(f"为 transformer 0 安装了 timing hooks")
    
    def get_step_config(self, args):
        """
        Generate step configuration for model switching.
        
        Args:
            args: Arguments object with steps attribute (list of step counts per model)
        
        Returns:
            total_step: Total number of inference steps
            step_config: Dictionary mapping step indices to model indices
        """
        assert len(args.steps) > 0, "steps must be non-empty"
        assert len(self.weight_folders) == len(args.steps), \
            f"Number of weight folders ({len(self.weight_folders)}) must match number of step configs ({len(args.steps)})"
        
        step_config = {
            "step": {},
            "name": {}
        }
        
        total_step = 0
        for index, model_step in enumerate(args.steps):
            for i in range(model_step):
                step_config["step"][total_step] = index
                total_step += 1
        
        for index, model_name in enumerate(self.weight_folders):
            step_config["name"][index] = model_name.split("/")[-1]
        
        return total_step, step_config
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = None,
        num_frames: int = 49,
        height: int = 480,
        width: int = 720,
        guidance_scale: float = 6.0,
        num_videos_per_prompt: int = 1,
        use_dynamic_cfg: bool = False,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        enable_profiling: bool = False,
        **kwargs
    ):
        """
        Generate video from text prompt using collaborative inference.
        
        Args:
            prompt: Text prompt(s) for video generation
            negative_prompt: Negative prompt(s)
            num_frames: Number of frames in the video
            height: Video height
            width: Video width
            guidance_scale: Guidance scale for classifier-free guidance
            num_videos_per_prompt: Number of videos to generate per prompt
            use_dynamic_cfg: Whether to use dynamic CFG
            prompt_embeds: Pre-computed prompt embeddings
            negative_prompt_embeds: Pre-computed negative prompt embeddings
            output_type: Output type ("pil", "np", "latent")
            enable_profiling: Whether to enable step-by-step profiling
            **kwargs: Additional arguments
        
        Returns:
            Generated video frames
        """
        # 创建callback函数用于profiling
        step_start_time = None
        current_step = 0
        num_steps = self.total_step
        latent_shape = None
        profiler = self.timing_profiler
        
        def step_callback(step, timestep, callback_kwargs):
            """每步结束后的回调，用于计时"""
            nonlocal step_start_time, current_step, latent_shape
            
            # 同步CUDA确保计时准确
            torch.cuda.synchronize()
            step_end_time = time.perf_counter()
            step_time = step_end_time - step_start_time
            
            # 获取latent维度
            if 'latents' in callback_kwargs and callback_kwargs['latents'] is not None:
                latent_shape = tuple(callback_kwargs['latents'].shape)
            
            # 获取模型名称
            model_name = self.weight_folders[0].split("/")[-1] if self.weight_folders else "Unknown"
            
            # 打印step计时 - CogVideoX传递的step是0-based
            profiler.print_step_timing(
                step,  # step是0-based
                num_steps,
                model_name,
                latent_shape if latent_shape else (1, 16, 21, height//8, width//8),
                step_time
            )
            
            # 准备下一步
            step_start_time = time.perf_counter()
            profiler.start_step()
            current_step = step
        
        # 如果启用profiling，在开始第一步之前记录初始状态
        if enable_profiling or self._profiling_enabled:
            profiler.start_step()
            step_start_time = time.perf_counter()
        
        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            num_inference_steps=self.total_step,
            guidance_scale=guidance_scale,
            num_videos_per_prompt=num_videos_per_prompt,
            use_dynamic_cfg=use_dynamic_cfg,
            generator=self.generator,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type=output_type,
            return_dict=True,
            callback_on_step_end=step_callback if (enable_profiling or self._profiling_enabled) else None,
            callback_on_step_end_tensor_inputs=["latents"],
            **kwargs
        )
        
        # 如果启用profiling，处理最后一步
        if (enable_profiling or self._profiling_enabled) and step_start_time is not None:
            torch.cuda.synchronize()
            step_end_time = time.perf_counter()
            step_time = step_end_time - step_start_time
            
            if latent_shape is None:
                latent_shape = (1, 16, 21, height//8, width//8)
            
            model_name = self.weight_folders[0].split("/")[-1] if self.weight_folders else "Unknown"
            profiler.print_step_timing(
                current_step,
                num_steps,
                model_name,
                latent_shape,
                step_time
            )
            
            # 输出总step时间
            total_step_time = profiler.get_total_step_time()
            print(f"\n所有Step总时间: {total_step_time:.2f}秒")
        
        return output.frames
    
    def _count_params(self, model):
        """Count parameters in a model"""
        return sum(p.numel() for p in model.parameters())
    
    def get_model_params(self):
        """
        Get parameter counts for all models.
        
        Returns:
            String describing model parameters
        """
        params_str = ""
        for index in range(len(self.pipe.transformers)):
            model_name = self.weight_folders[index].split("/")[-1]
            cur_transformer = self._count_params(self.pipe.transformers[index])
            params_str += f" {model_name}: {(cur_transformer/1e6):.1f}M"
        
        params_text_enc = self._count_params(self.pipe.text_encoder)
        params_vae_dec = self._count_params(self.pipe.vae.decoder)
        
        return_str = params_str + f" TextEnc {(params_text_enc/1e6):.1f}M; VAE {(params_vae_dec/1e6):.1f}M"
        return return_str