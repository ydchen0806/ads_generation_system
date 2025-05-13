#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration file for the advertisement generation system.
"""

import os
import torch

class Config:
    """Configuration class for the ad generation system."""
    
    def __init__(self):
        # Paths
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.base_dir, "output")
        self.debug_dir = os.path.join(self.output_dir, "debug")
        self.fallback_image_path = os.path.join(self.base_dir, "assets", "fallback.png")
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Qwen2.5-VL model configuration
        # self.qwen_model_path = "/h3cstore_ns/ydchen/code/SubjectGenius/model/Qwen2.5-VL-7B-Instruct"
        self.use_flash_attention = True
        self.min_pixels = 256 * 28 * 28  # Minimum number of pixels for visual tokens
        self.max_pixels = 1280 * 28 * 28  # Maximum number of pixels for visual tokens
        self.max_new_tokens = 1024
        self.temperature = 0.7
        self.top_p = 0.9
        
        # FLUX and OmniControl model paths and settings
        self.flux_model_path = "/data/ydchen/VLP/SubjectGenius/model/FLUX.1-schnell"
        self.omnicontrol_path = "/data/ydchen/VLP/SubjectGenius/model/OminiControl"
        self.omnicontrol_weight_name = "omini/subject_512.safetensors"
        
        # OmniControl parameters
        self.position_delta = (0, 32)  # 允许物体位置的轻微变化
        self.image_size = (512, 512)
        self.debug_dir = os.path.join(self.output_dir, "debug")  # 添加调试目录
        self.num_inference_steps = 8
        self.guidance_scale = 7.5
        self.default_seed = 42
        self.negative_prompt = "low quality, blurry, distorted, deformed, disfigured"
        
        # Wan model configuration
        self.wan_model_path = "/h3cstore_ns/ydchen/code/wan_2_1/Wan2.1/model/Wan2.1-FLF2V-14B-720P"
        self.wan_script_path = "/h3cstore_ns/ydchen/code/wan_2_1/Wan2.1/generate.py"
        self.use_distributed = True
        self.gpus_per_node = 8  # Number of GPUs to use for distributed generation
        self.pip_requirements = [
            "-r /h3cstore_ns/ydchen/code/wan_2_1/Wan2.1/requirements.txt",
            "opencv-python==4.5.5.64 opencv-contrib-python==4.5.5.64",
            "flash_attn==2.6.3 --no-build-isolation",
            "\"xfuser>=0.4.1\""
        ]
        self.qwen_model_path = "/h3cstore_ns/ydchen/code/SubjectGenius/model/Qwen2.5-VL-7B-Instruct"  # 模型路径
        self.temp_dir = "/tmp"  # 临时目录，用于保存图像文件
        self.max_tokens = 512  # 生成的最大 token 数
        # Output video settings
        self.video_resolution = (1280, 720)
        self.video_fps = 30
        
        # Prompt engineering
        self.director_system_prompt = """
        You are an award-winning creative director who specializes in creating compelling advertising campaigns. 
        Your expertise is crafting marketing narratives that tell powerful stories about products through carefully 
        planned visual sequences.

        You excel at:
        1. Identifying the most marketable aspects of any product
        2. Creating dramatic narrative arcs that engage viewers emotionally
        3. Designing visually distinctive scenes that highlight product features
        4. Balancing aspirational imagery with authentic product representation
        5. Incorporating brand identity and marketing psychology into visual storytelling

        Your goal is to create keyframe descriptions that would guide a world-class production team to create 
        a marketing video that drives consumer interest and conversion.
        """
        
        self.transition_system_prompt = """
        You are an elite commercial director who specializes in creating seamless, emotionally impactful 
        transitions for high-end product advertisements. Your transitions are known for their cinematic quality, 
        technical precision, and ability to maintain narrative continuity.

        You have expertise in:
        1. Camera movement language (pans, zooms, dollies, etc.)
        2. Visual effects techniques (dissolves, morphs, parallax, etc.)
        3. Timing and pacing for emotional impact
        4. Maintaining product focus through scene changes
        5. Creating visual poetry that enhances the marketing message

        Your goal is to create transition descriptions that would guide a professional editor to create 
        seamless, beautiful connections between scenes that enhance the product story and maintain viewer engagement.
        """
        
        # Create necessary directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.debug_dir, exist_ok=True)