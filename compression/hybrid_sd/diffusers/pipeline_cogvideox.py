# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The HuggingFace Team. All rights reserved.
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

"""
Hybrid CogVideoX Pipeline for collaborative inference
Supports switching between multiple transformers (e.g., CogVideoX-5B and CogVideoX-2B) during denoising
"""

import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Union
import os
import math

import torch
from packaging import version

import diffusers
from diffusers import CogVideoXPipeline
from diffusers.models import CogVideoXTransformer3DModel
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from diffusers.utils import logging

logger = logging.get_logger(__name__)


class HybridCogVideoXPipeline(CogVideoXPipeline):
    r"""
    Pipeline for text-to-video generation using CogVideoX with collaborative inference.
    
    This pipeline extends CogVideoXPipeline to support switching between multiple transformers
    (e.g., CogVideoX-5B and CogVideoX-2B) during the denoising process.
    
    The pipeline inherits all methods from CogVideoXPipeline and adds:
    - Multiple transformer support
    - Step configuration for model switching
    - Rotary positional embeddings handling for different model types
    """
    
    def __init__(
        self,
        transformer=None,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        scheduler=None
    ):
        # Call parent __init__ with all required arguments
        super().__init__(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler
        )
        
        # Multiple transformers for collaborative inference
        self.transformers = None  # List of transformers [large_model, small_model]
        self.step_config = None   # Step configuration dict
        self.scheduler_configs = None  # List of scheduler configs for each model
        
    def set_transformers(self, transformers: List[CogVideoXTransformer3DModel]):
        """
        Set multiple transformers for collaborative inference.
        
        Args:
            transformers: List of CogVideoXTransformer3DModel instances
        """
        self.transformers = transformers
        # Keep the first transformer as the default for compatibility
        if transformers and len(transformers) > 0:
            self.transformer = transformers[0]
    
    def set_step_config(self, step_config: Dict[str, Any]):
        """
        Set step configuration for model switching.
        
        Args:
            step_config: Dictionary with keys:
                - "step": Dict mapping step index to model index
                - "name": Dict mapping model index to model name
        """
        self.step_config = step_config
    
    def set_scheduler_configs(self, scheduler_configs: List[Dict[str, Any]]):
        """
        Set scheduler configurations for each model.
        
        Args:
            scheduler_configs: List of scheduler config dictionaries, one for each model
        """
        self.scheduler_configs = scheduler_configs
    
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Get T5 prompt embeddings with proper device handling.
        
        This method ensures device is properly converted to torch.device object
        before processing, fixing the device mismatch issue.
        """
        # Ensure device is a torch.device object (not a string)
        if device is None:
            device = self._execution_device
        
        if isinstance(device, str):
            device = torch.device(device)
        
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        # Ensure text_input_ids is moved to the correct device
        # Convert device to torch.device if it's a string
        if isinstance(device, str):
            device = torch.device(device)
        
        # Get the actual device of text_encoder
        text_encoder_device = next(self.text_encoder.parameters()).device
        
        # Move text_input_ids to the same device as text_encoder
        text_input_ids = text_input_ids.to(text_encoder_device)
        prompt_embeds = self.text_encoder(text_input_ids)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 226,
    ):
        """
        Encode the prompt into text embeddings.
        
        This method ensures device is properly converted to torch.device object
        before calling the parent's encode_prompt method.
        """
        # Ensure device is a torch.device object (not a string)
        if device is None:
            device = self._execution_device
        
        if isinstance(device, str):
            device = torch.device(device)
        
        dtype = dtype or self.text_encoder.dtype

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            if isinstance(prompt, list):
                batch_size = len(prompt)
                negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            else:
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            
            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6.0,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        **kwargs,
    ):
        """
        The call function to the pipeline for generation with collaborative inference.
        
        This method extends the parent __call__ method to support switching between
        multiple transformers during denoising based on step_config.
        """
        
        # Check if we're using collaborative inference
        use_hybrid = (
            self.transformers is not None 
            and len(self.transformers) > 1 
            and self.step_config is not None
        )
        
        if not use_hybrid:
            # Fall back to standard CogVideoXPipeline behavior
            logger.warning("Hybrid mode not enabled, using standard pipeline")
            return super().__call__(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
                use_dynamic_cfg=use_dynamic_cfg,
                num_videos_per_prompt=num_videos_per_prompt,
                eta=eta,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
                **kwargs,
            )
        
        # Override the transformer call in the parent method
        # We need to intercept the denoising loop
        return self._hybrid_call(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            num_videos_per_prompt=num_videos_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type=output_type,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            **kwargs,
        )
    
    def _hybrid_call(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6.0,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        **kwargs,
    ):
        """
        Internal method for hybrid collaborative inference.
        """
        # Set internal state
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        
        # Get device - prefer CUDA if models are on CUDA, otherwise use _execution_device
        device = self._execution_device
        if isinstance(device, str):
            device = torch.device(device)
        
        # If models are on CUDA, ensure device is CUDA
        # Check where the transformer is located
        if self.transformers and len(self.transformers) > 0:
            transformer_device = next(self.transformers[0].parameters()).device
            if transformer_device.type == 'cuda':
                device = transformer_device
        elif hasattr(self, 'transformer') and self.transformer is not None:
            transformer_device = next(self.transformer.parameters()).device
            if transformer_device.type == 'cuda':
                device = transformer_device
        
        # 1. Check inputs
        if height is None:
            height = self.vae.config.sample_height
        if width is None:
            width = self.vae.config.sample_width
        if num_frames is None:
            num_frames = self.transformer.config.sample_frames
        
        # 2. Encode prompt
        do_classifier_free_guidance = guidance_scale > 1.0
        
        if prompt_embeds is None:
            # Ensure device is a torch.device object (not a string)
            if device is None:
                device = self._execution_device
            if isinstance(device, str):
                device = torch.device(device)
            
            # Call encode_prompt which will handle device conversion
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                device=device,
            )
        
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        
        # 3. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        self._num_timesteps = len(timesteps)
        
        # 4. Prepare latents (match parent pipeline logic including temporal padding)
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        patch_size_t = getattr(self.transformer.config, "patch_size_t", None)
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames = num_frames + additional_frames * self.vae_scale_factor_temporal
        
        if latents is None:
            # Ensure device matches generator's device if generator is provided
            if generator is not None:
                if hasattr(generator, "device"):
                    generator_device = generator.device
                    if generator_device.type == "cuda":
                        device = generator_device
                elif isinstance(generator, list) and len(generator) > 0:
                    if hasattr(generator[0], "device"):
                        generator_device = generator[0].device
                        if generator_device.type == "cuda":
                            device = generator_device
            
            if isinstance(device, str):
                device = torch.device(device)
            
            latents = super().prepare_latents(
                batch_size=num_videos_per_prompt,
                num_channels_latents=self.transformer.config.in_channels,
                num_frames=num_frames,
                height=height,
                width=width,
                dtype=prompt_embeds.dtype,
                device=device,
                generator=generator,
            )
        else:
            latents = latents.to(device=device)
        
        logger.info(
            "Initial latents - min: %.4f, max: %.4f, mean: %.4f, std: %.4f, dtype: %s, shape: %s",
            latents.min().item(),
            latents.max().item(),
            latents.mean().item(),
            latents.std().item(),
            latents.dtype,
            tuple(latents.shape),
        )
        
        # 5. Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        # 6. Prepare rotary embeddings for each transformer
        # This is critical: 5B uses rotary embeddings, 2B doesn't
        # NOTE: We must use the ACTUAL latent dimensions after VAE encoding,
        # not the input video dimensions. The latents are already temporal-compressed.
        # Use latents.size(1) for the actual number of frames in latent space.
        image_rotary_embs = []
        if self.transformers and len(self.transformers) > 0:
            # Get actual latent dimensions from latents tensor
            # latents shape: (B, T_latent, C, H_latent, W_latent)
            actual_latent_frames = latents.size(1)  # Actual frames after temporal compression
            actual_latent_height = latents.size(3)  # Actual height in latent space
            actual_latent_width = latents.size(4)   # Actual width in latent space
            
            for transformer in self.transformers:
                if transformer.config.use_rotary_positional_embeddings:
                    # Use ACTUAL latent dimensions from the latents tensor
                    # This matches what the parent class does: latents.size(1)
                    image_rotary_emb = super()._prepare_rotary_positional_embeddings(
                        height=height,  # Use output height for grid calculation
                        width=width,    # Use output width for grid calculation
                        num_frames=actual_latent_frames,  # Use ACTUAL latent frames
                        device=device,
                    )
                else:
                    image_rotary_emb = None
                image_rotary_embs.append(image_rotary_emb)
        else:
            # Fallback: use provided dimensions
            if hasattr(self, 'transformer') and self.transformer is not None:
                if self.transformer.config.use_rotary_positional_embeddings:
                    # Use actual latent frames if latents are available
                    if latents is not None:
                        actual_latent_frames = latents.size(1)
                    else:
                        actual_latent_frames = num_frames
                    image_rotary_embs.append(super()._prepare_rotary_positional_embeddings(
                        height=height,
                        width=width,
                        num_frames=actual_latent_frames,
                        device=device,
                    ))
                else:
                    image_rotary_embs.append(None)
            else:
                image_rotary_embs.append(None)
        
        # 7. Denoising loop with model switching
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        
        # For CogVideoXDPMScheduler, we need to track old_pred_original_sample
        # Initialize as None (same as parent class CogVideoXPipeline)
        from diffusers import CogVideoXDPMScheduler
        old_pred_original_sample = None
        original_latents_dtype = latents.dtype  # Store original dtype
        
        # Log scheduler type for debugging
        logger.info(f"Scheduler type: {type(self.scheduler).__name__}, is CogVideoXDPMScheduler: {isinstance(self.scheduler, CogVideoXDPMScheduler)}")
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self._interrupt:
                    continue
                
                # Check latents at the beginning of each step for NaN/Inf
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    logger.error(f"Step {i}/{num_inference_steps}: latents contains NaN/Inf at the start of step!")
                    logger.error(f"latents - min: {latents.min().item():.6f}, max: {latents.max().item():.6f}, mean: {latents.mean().item():.6f}")
                    if i > 0:
                        logger.error(f"Previous step was {i-1}, model was {self.step_config['name'][self.step_config['step'][i-1]]}")
                    raise ValueError(f"NaN/Inf detected in latents at step {i}")
                
                # Expand latents for classifier-free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                # Check latent_model_input after scaling
                if torch.isnan(latent_model_input).any() or torch.isinf(latent_model_input).any():
                    logger.error(f"Step {i}/{num_inference_steps}: latent_model_input contains NaN/Inf after scale_model_input!")
                    logger.error(f"latent_model_input - min: {latent_model_input.min().item():.6f}, max: {latent_model_input.max().item():.6f}")
                    logger.error(f"latents before scaling - min: {latents.min().item():.6f}, max: {latents.max().item():.6f}")
                    logger.error(f"timestep t: {t.item()}")
                    raise ValueError(f"NaN/Inf detected in latent_model_input after scaling at step {i}")
                
                # Broadcast timestep
                # Ensure timestep has the correct dtype (long/int64) for time embedding
                timestep = t.expand(latent_model_input.shape[0])
                
                # Select transformer based on step_config
                model_index = self.step_config["step"][i]
                model_name = self.step_config["name"][model_index]
                selected_transformer = self.transformers[model_index]
                image_rotary_emb = image_rotary_embs[model_index]
                
                # Check if model switched - if so, update scheduler config and reset old_pred_original_sample
                if i > 0:
                    prev_model_index = self.step_config["step"][i - 1]
                    if prev_model_index != model_index:
                        logger.info(f"=== MODEL SWITCH at step {i}: {self.step_config['name'][prev_model_index]} -> {model_name} ===")
                        logger.info(f"Scheduler type: {type(self.scheduler).__name__}, is CogVideoXDPMScheduler: {isinstance(self.scheduler, CogVideoXDPMScheduler)}")
                        
                        # Update scheduler config if different models have different configs
                        if self.scheduler_configs is not None and model_index < len(self.scheduler_configs):
                            new_scheduler_config = self.scheduler_configs[model_index]
                            current_config = self.scheduler.config
                            
                            # Check if snr_shift_scale needs to be updated (key difference between 5B and 2B)
                            if "snr_shift_scale" in new_scheduler_config:
                                new_snr_shift_scale = new_scheduler_config["snr_shift_scale"]
                                current_snr_shift_scale = getattr(current_config, "snr_shift_scale", None)
                                
                                if new_snr_shift_scale != current_snr_shift_scale:
                                    logger.info(f"Updating scheduler snr_shift_scale: {current_snr_shift_scale} -> {new_snr_shift_scale}")
                                    # Update scheduler config
                                    self.scheduler.config.snr_shift_scale = new_snr_shift_scale
                                    # Re-initialize scheduler to apply the change
                                    # For DDIM scheduler, we need to update the internal state
                                    from diffusers import CogVideoXDDIMScheduler
                                    if isinstance(self.scheduler, CogVideoXDDIMScheduler):
                                        # Create new scheduler with updated config
                                        updated_config = self.scheduler.config.copy()
                                        updated_config["snr_shift_scale"] = new_snr_shift_scale
                                        self.scheduler = CogVideoXDDIMScheduler.from_config(updated_config)
                                        # Restore timesteps using the existing timesteps variable
                                        # timesteps and num_inference_steps are already set before the denoising loop
                                        self.scheduler.set_timesteps(num_inference_steps, device=latents.device)
                                        logger.info(f"Scheduler updated with snr_shift_scale={new_snr_shift_scale}, timesteps restored (num_inference_steps={num_inference_steps})")
                                    
                        if isinstance(self.scheduler, CogVideoXDPMScheduler):
                            logger.info(f"Resetting old_pred_original_sample (was: {old_pred_original_sample is not None})")
                            old_pred_original_sample = None
                        else:
                            logger.info(f"Scheduler is {type(self.scheduler).__name__}, no old_pred_original_sample to reset")
                
                logger.info(f"Step {i}/{num_inference_steps}: Using {model_name} (index {model_index})")
                
                # Debug: Log shapes (first step only)
                if i == 0:
                    logger.debug(f"latent_model_input shape: {latent_model_input.shape}")
                    logger.debug(f"prompt_embeds shape: {prompt_embeds.shape}")
                    logger.debug(f"timestep shape: {timestep.shape}")
                
                # Check latents for NaN/Inf before transformer prediction
                if torch.isnan(latent_model_input).any() or torch.isinf(latent_model_input).any():
                    logger.error(f"Step {i}/{num_inference_steps}: latent_model_input contains NaN/Inf before transformer!")
                    logger.error(f"latent_model_input - min: {latent_model_input.min().item():.6f}, max: {latent_model_input.max().item():.6f}, mean: {latent_model_input.mean().item():.6f}")
                    raise ValueError(f"NaN/Inf detected in latent_model_input at step {i}")
                
                # Predict noise with selected transformer
                # Use cache_context for conditional/unconditional guidance
                # Use autocast to handle dtype conversion (time_proj outputs float32, but model weights are float16)
                with selected_transformer.cache_context("cond_uncond" if do_classifier_free_guidance else None):
                    with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
                        noise_pred = selected_transformer(
                            hidden_states=latent_model_input,
                            encoder_hidden_states=prompt_embeds,
                            timestep=timestep,
                            image_rotary_emb=image_rotary_emb,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                noise_pred = noise_pred.float()
                logger.info(
                    "Step %d/%d: noise_pred raw - min: %.4f, max: %.4f, mean: %.4f, std: %.4f",
                    i,
                    num_inference_steps,
                    noise_pred.min().item(),
                    noise_pred.max().item(),
                    noise_pred.mean().item(),
                    noise_pred.std().item(),
                )
                
                # Check noise_pred for NaN/Inf immediately after transformer
                if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
                    logger.error(f"Step {i}/{num_inference_steps}: noise_pred contains NaN/Inf after transformer!")
                    logger.error(f"noise_pred - min: {noise_pred.min().item():.6f}, max: {noise_pred.max().item():.6f}, mean: {noise_pred.mean().item():.6f}")
                    raise ValueError(f"NaN/Inf detected in noise_pred at step {i}")
                
                # Perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (
                            1
                            - math.cos(
                                math.pi
                                * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0
                            )
                        )
                        / 2
                    )
                
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self._guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )
                    logger.info(
                        "Step %d/%d: noise_pred guided - min: %.4f, max: %.4f, mean: %.4f, std: %.4f",
                        i,
                        num_inference_steps,
                        noise_pred.min().item(),
                        noise_pred.max().item(),
                        noise_pred.mean().item(),
                        noise_pred.std().item(),
                    )
                    
                    # Check noise_pred after guidance
                    if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
                        logger.error(f"Step {i}/{num_inference_steps}: noise_pred contains NaN/Inf after guidance!")
                        logger.error(f"guidance_scale: {self._guidance_scale}, noise_pred - min: {noise_pred.min().item():.6f}, max: {noise_pred.max().item():.6f}")
                        raise ValueError(f"NaN/Inf detected in noise_pred after guidance at step {i}")
                
                # Compute previous noisy sample
                # Handle CogVideoXDPMScheduler differently (same as parent class CogVideoXPipeline)
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    # Standard scheduler step
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                    )[0]
                else:
                    # CogVideoXDPMScheduler: use timesteps[i-1] if i > 0, else None (same as parent class)
                    # This matches the parent class implementation exactly
                    timestep_back = timesteps[i - 1] if i > 0 else None
                    
                    # Check latents before scheduler.step
                    if torch.isnan(latents).any() or torch.isinf(latents).any():
                        logger.error(f"Step {i}/{num_inference_steps}: latents contains NaN/Inf before scheduler.step!")
                        raise ValueError(f"NaN/Inf detected in latents before scheduler.step at step {i}")
                    
                    # CogVideoXDPMScheduler.step signature: (model_output, old_pred_original_sample, timestep, timestep_back, sample, ...)
                    # Note: timestep should be int (not tensor), and timestep_back can be None for first step
                    try:
                        latents, old_pred_original_sample = self.scheduler.step(
                            noise_pred,
                            old_pred_original_sample,
                            t,
                            timestep_back,
                            latents,
                            **extra_step_kwargs,
                            return_dict=False,
                        )
                    except Exception as e:
                        logger.error(f"Step {i}/{num_inference_steps}: Error in scheduler.step: {e}")
                        logger.error(f"noise_pred shape: {noise_pred.shape}, dtype: {noise_pred.dtype}")
                        logger.error(f"latents shape: {latents.shape}, dtype: {latents.dtype}")
                        logger.error(f"t: {t}")
                        logger.error(f"timestep_back: {timestep_back}")
                        logger.error(f"old_pred_original_sample: {old_pred_original_sample}")
                        raise
                    
                    # Check latents after scheduler.step for NaN/Inf
                    if torch.isnan(latents).any() or torch.isinf(latents).any():
                        logger.error(f"Step {i}/{num_inference_steps}: latents contains NaN/Inf after scheduler.step!")
                        logger.error(f"latents - min: {latents.min().item():.6f}, max: {latents.max().item():.6f}, mean: {latents.mean().item():.6f}")
                        logger.error(f"noise_pred - min: {noise_pred.min().item():.6f}, max: {noise_pred.max().item():.6f}")
                        logger.error(f"t: {t}, timestep_back: {timestep_back}")
                        raise ValueError(f"NaN/Inf detected in latents after scheduler.step at step {i}")
                
                # Convert latents back to prompt_embeds dtype (same as parent class)
                latents = latents.to(prompt_embeds.dtype)
                logger.info(
                    "Step %d/%d: latents after step - min: %.4f, max: %.4f, mean: %.4f, std: %.4f",
                    i,
                    num_inference_steps,
                    latents.min().item(),
                    latents.max().item(),
                    latents.mean().item(),
                    latents.std().item(),
                )
                
                # Callback
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_on_step_end(i, t, callback_kwargs)
                
                # Update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        
        # 8. Decode latents to video
        # Use parent class decode_latents method
        if output_type == "latent":
            video = latents
        else:
            if additional_frames > 0:
                latents = latents[:, additional_frames:]
            # Check latents for NaN or Inf values before decoding
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                logger.error("Latents contain NaN or Inf values before VAE decoding! This should not happen.")
                logger.error(f"Latents - min: {latents.min().item():.6f}, max: {latents.max().item():.6f}, mean: {latents.mean().item():.6f}")
                raise ValueError("Latents contain NaN/Inf values before VAE decoding. Check the denoising loop.")
            
            # Log latents statistics for debugging (use INFO level so it's always shown)
            logger.info(f"Latents before decode - min: {latents.min().item():.4f}, "
                       f"max: {latents.max().item():.4f}, "
                       f"mean: {latents.mean().item():.4f}, "
                       f"std: {latents.std().item():.4f}, "
                       f"dtype: {latents.dtype}, shape: {latents.shape}")
            
            # Ensure latents are on the same device as VAE for decoding
            # If VAE is on a different GPU, move latents to VAE's device
            vae_device = next(self.vae.parameters()).device
            if latents.device != vae_device:
                logger.info(f"Moving latents from {latents.device} to VAE device {vae_device} for decoding")
                latents = latents.to(vae_device)
            
            # Ensure latents are in the correct dtype for VAE decoding
            # VAE weights are float16, so we need to convert latents to float16
            # But first, let's check what dtype the parent's decode_latents expects
            # The parent class should handle this, but we need to ensure latents are properly scaled
            # Use parent class decode_latents which handles scaling and dtype conversion
            # The parent class should handle dtype conversion internally
            try:
                video = self.decode_latents(latents)
            except RuntimeError as e:
                if "Input type" in str(e) and "bias type" in str(e):
                    # dtype mismatch error - convert latents to VAE dtype
                    logger.warning(f"VAE dtype mismatch detected. Converting latents to VAE dtype.")
                    vae_dtype = next(self.vae.parameters()).dtype
                    logger.info(f"VAE dtype: {vae_dtype}, latents dtype: {latents.dtype}")
                    # Convert latents to VAE dtype before decoding
                    latents_converted = latents.to(dtype=vae_dtype)
                    video = self.decode_latents(latents_converted)
                else:
                    raise
            
            # Check decoded video for valid values
            if isinstance(video, torch.Tensor):
                if torch.isnan(video).any() or torch.isinf(video).any():
                    logger.warning("Decoded video contains NaN or Inf values!")
                logger.info(f"Video after decode (tensor) - min: {video.min().item():.4f}, "
                           f"max: {video.max().item():.4f}, "
                           f"mean: {video.mean().item():.4f}, "
                           f"std: {video.std().item():.4f}, shape: {video.shape}")
            elif isinstance(video, np.ndarray):
                logger.info(f"Video after decode (numpy) - min: {video.min():.4f}, "
                           f"max: {video.max():.4f}, "
                           f"mean: {video.mean():.4f}, "
                           f"std: {video.std():.4f}, shape: {video.shape}, dtype: {video.dtype}")
            else:
                logger.info(f"Video after decode - type: {type(video)}, len: {len(video) if hasattr(video, '__len__') else 'N/A'}")
        
        # 9. Convert to output format (use parent class video_processor)
        # postprocess_video expects keyword argument 'video' and returns a list of videos
        # Each video is a list of PIL images (for output_type="pil")
        if output_type == "pil":
            video = self.video_processor.postprocess_video(video=video, output_type="pil")
        elif output_type == "np":
            video = self.video_processor.postprocess_video(video=video, output_type="np")
        
        if not return_dict:
            return (video,)
        
        # Return in CogVideoXPipelineOutput format
        from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput
        return CogVideoXPipelineOutput(frames=video)

