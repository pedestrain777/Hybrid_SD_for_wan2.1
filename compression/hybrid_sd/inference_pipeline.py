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
from .diffusers.pipeline_wan import HybridWanPipeline
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
            # Use the same device as main device (single GPU mode)
            self.vae_device = self.device
            print(f"VAE device set to main device: {self.vae_device}")
        self.seed = seed
        self.args = args
        self.pipe = None
        self.generator = None
    
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
        # Import Wan components
        from transformers import UMT5EncoderModel, AutoTokenizer
        from diffusers import AutoencoderKLWan, WanTransformer3DModel
        from diffusers import WanPipeline
        
        # 1. Load shared Text Encoder (UMT5)
        print(f"Loading Text Encoder (UMT5) from {self.weight_folders[0]}")
        try:
            text_encoder = UMT5EncoderModel.from_pretrained(
                self.weight_folders[0], subfolder="text_encoder"
            ).to(self.device, dtype=torch.float16).requires_grad_(False)
            print(f"✅ UMT5 Text Encoder loaded successfully")
        except Exception as e:
            print(f"❌ Text Encoder loading failed: {e}")
            raise

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.weight_folders[0], subfolder="tokenizer"
            )
            print(f"✅ Tokenizer loaded successfully")
        except Exception as e:
            print(f"❌ Tokenizer loading failed: {e}")
            raise
        
        # 2. Load shared VAE on separate GPU if available
        print(f"Loading VAE from {self.weight_folders[0]} on device {self.vae_device}")
        try:
            vae = AutoencoderKLWan.from_pretrained(
                self.weight_folders[0], subfolder="vae"
            ).to(self.vae_device, dtype=torch.float16).requires_grad_(False)
            # Enable VAE memory optimizations to reduce memory usage
            vae.enable_slicing()
            vae.enable_tiling()
            print(f"✅ AutoencoderKLWan loaded successfully (device: {self.vae_device})")
            print(f"   - VAE slicing: enabled")
            print(f"   - VAE tiling: enabled")
        except Exception as e:
            print(f"❌ VAE loading failed: {e}")
            raise
        
        # 3. Load multiple transformers
        print(f"Loading {len(self.weight_folders)} transformers...")
        transformers = []
        for path in self.weight_folders:
            print(f"  Loading transformer from {path}")
            try:
                # Load on single GPU (changed from device_map="auto")
                transformer = WanTransformer3DModel.from_pretrained(
                    path, subfolder="transformer", torch_dtype=torch.float16
                ).to(self.device).requires_grad_(False)
                # Ensure all modules are in float16
                transformer = transformer.half()
                print(f"  ✅ WanTransformer3DModel loaded successfully (single GPU: {self.device})")
            except Exception as e:
                print(f"  ❌ Transformer loading failed: {e}")
                raise
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
        print("Creating HybridWanPipeline...")
        # Create base pipeline with the first transformer to ensure proper initialization
        base_pipeline = WanPipeline.from_pretrained(
            self.weight_folders[0],
            text_encoder=text_encoder,
            vae=vae,
            tokenizer=tokenizer,
            transformer=transformers[0],
            torch_dtype=torch.float16
        )
        print(f"✅ WanPipeline base pipeline created successfully")
        
        # Convert to HybridWanPipeline
        # Use the transformer from base_pipeline to ensure it's properly configured
        self.pipe = HybridWanPipeline(
            transformer=base_pipeline.transformer,  # Use pipeline's transformer (properly configured)
            vae=base_pipeline.vae,
            text_encoder=base_pipeline.text_encoder,
            tokenizer=base_pipeline.tokenizer,
            scheduler=base_pipeline.scheduler,
            # Wan-specific parameters (verified to exist in WanPipeline.__init__)
            transformer_2=None,        # Second transformer (for dual-transformer architecture)
            boundary_ratio=None,       # Boundary ratio (default None, let Wan handle automatically)
            expand_timesteps=False     # Whether to expand timesteps (default False)
        )
        print(f"✅ HybridWanPipeline created successfully")
        print(f"   - transformer_2: {self.pipe.transformer_2 is not None}")
        print(f"   - boundary_ratio: {self.pipe.boundary_ratio}")
        print(f"   - expand_timesteps: {self.pipe.expand_timesteps}")
        
        # 6. Set multiple transformers
        # Replace the first transformer with the one from pipeline (if needed)
        # and add additional transformers
        # Ensure all transformers are properly loaded with the same configuration
        pipeline_transformers = [base_pipeline.transformer]  # First transformer from pipeline
        for i in range(1, len(transformers)):
            # Load additional transformers on single GPU (changed from device_map="auto")
            try:
                additional_transformer = WanTransformer3DModel.from_pretrained(
                    self.weight_folders[i],
                    subfolder="transformer",
                    torch_dtype=torch.float16
                ).to(self.device).requires_grad_(False)
                print(f"  ✅ Additional Transformer #{i} loaded successfully (single GPU: {self.device})")
            except Exception as e:
                print(f"  ❌ Additional Transformer #{i} loading failed: {e}")
                raise
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
        self.pipe.set_hybrid_mask_config({
            "ema_alpha": getattr(self.args, "hybrid_ema_alpha", 0.7),
            "temporal_top_ratio": getattr(self.args, "hybrid_temporal_top_ratio", 0.30),
            "spatial_top_ratio": getattr(self.args, "hybrid_spatial_top_ratio", 0.20),
            "temporal_dilate": getattr(self.args, "hybrid_temporal_dilate", 1),
            "spatial_dilate": getattr(self.args, "hybrid_spatial_dilate", 3),
            "relative_diff": getattr(self.args, "hybrid_relative_diff", True),
        })
        
        # 9. Set generator
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        
        # 10. Configure scheduler if needed
        # Wan uses FlowMatchEulerDiscreteScheduler, usually no need to modify
        # If custom scheduler is needed:
        if hasattr(self.args, 'use_custom_scheduler') and self.args.use_custom_scheduler:
            from diffusers import FlowMatchEulerDiscreteScheduler
            self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )
            print(f"✅ Using custom FlowMatchEulerDiscreteScheduler")
        # Note: Wan does not use DPM solver, uses Flow Matching
        
        # 11. Disable progress bar if needed
        if hasattr(self.pipe, 'set_progress_bar_config'):
            self.pipe.set_progress_bar_config(disable=True)
    
    def get_step_config(self, args):
        """
        三阶段配置：
        - large:  只用大模型
        - hybrid: 小模型 full + 大模型 full + mask 融合
        - small:  只用小模型

        返回:
            total_step
            step_config = {
                "step": {step_idx: primary_model_index},   # hybrid阶段也用large作为primary
                "mode": {step_idx: "large"/"hybrid"/"small"},
                "name": {0: large_name, 1: small_name},
                "large_index": 0,
                "small_index": 1,
            }
        """
        assert len(self.weight_folders) == 2, "当前实现只支持两个模型：[large, small]"

        if hasattr(args, "stage_steps") and args.stage_steps is not None:
            stage_steps = args.stage_steps
        else:
            stage_steps = args.steps

        assert len(stage_steps) == 3, \
            f"现在必须传 3 段 steps，例如 [10,25,15]，当前得到: {stage_steps}"

        large_steps, hybrid_steps, small_steps = stage_steps

        step_config = {
            "step": {},
            "mode": {},
            "name": {},
            "large_index": 0,
            "small_index": 1,
        }

        total_step = 0

        for _ in range(large_steps):
            step_config["step"][total_step] = 0
            step_config["mode"][total_step] = "large"
            total_step += 1

        for _ in range(hybrid_steps):
            step_config["step"][total_step] = 0
            step_config["mode"][total_step] = "hybrid"
            total_step += 1

        for _ in range(small_steps):
            step_config["step"][total_step] = 1
            step_config["mode"][total_step] = "small"
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
            **kwargs: Additional arguments
        
        Returns:
            Generated video frames
        """
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
            **kwargs
        )
        
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