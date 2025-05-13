#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Connector for Wan model for video generation.
"""

import os
import sys
import logging
import tempfile
import torch
import numpy as np
import random
import subprocess
from PIL import Image
import shutil
from datetime import datetime
import sys
sys.path.append('/h3cstore_ns/ydchen/code/wan_2_1/ads_generation_system')
# Import Wan modules
import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WanModel:
    """Connector for the Wan model for video generation."""
    
    def __init__(self, config):
        """Initialize the Wan model connector.
        
        Args:
            config: Configuration containing model paths and parameters
        """
        self.config = config
        self.task = "flf2v-14B"  # First-Last-Frame to Video task
        self.size = "1280*720"   # Default video resolution
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model path and temporary directory
        self.model_path = self.config.wan_model_path
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {self.temp_dir}")
        
        # Configure default parameters
        self.sample_steps = 40
        self.sample_shift = 16.0
        self.guide_scale = 5.0
        self.frame_num = 81  # Default frame number for Wan
        self.sample_solver = "unipc"  # Default solver
        self.base_seed = random.randint(0, sys.maxsize)
        
        logger.info(f"Initialized Wan connector for task: {self.task}")
    
    def _save_image_for_wan(self, image, filename):
        """Save a PIL image to disk for Wan input.
        
        Args:
            image: PIL Image to save
            filename: Destination filename
            
        Returns:
            str: Full path to saved image
        """
        file_path = os.path.join(self.temp_dir, filename)
        image.save(file_path)
        return file_path
    
    def generate_transition(self, first_frame, last_frame, transition_prompt):
        """Generate a video transition between two keyframes.
        
        Args:
            first_frame: PIL Image of the first keyframe
            last_frame: PIL Image of the last keyframe
            transition_prompt: Text description of the transition
            
        Returns:
            str: Path to the generated video file
        """
        logger.info(f"Generating video transition with Wan {self.task}...")
        
        try:
            # Save keyframes to temporary directory
            first_frame_path = self._save_image_for_wan(first_frame, "first_frame.png")
            last_frame_path = self._save_image_for_wan(last_frame, "last_frame.png")
            
            logger.info(f"Saved keyframes to temporary files")
            logger.info(f"Transition prompt: {transition_prompt}")
            
            # Create a unique ID for this generation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = f"{timestamp}_{random.randint(1000, 9999)}"
            
            # Output video path
            output_video = os.path.join(self.temp_dir, f"transition_{unique_id}.mp4")
            
            # Prepare command for running Wan
            command = [
                "python", self.config.wan_script_path,
                "--task", self.task,
                "--size", self.size,
                "--ckpt_dir", self.model_path,
                "--save_file", output_video,
                "--first_frame", first_frame_path,
                "--last_frame", last_frame_path,
                "--prompt", transition_prompt,
                "--sample_steps", str(self.sample_steps),
                "--sample_shift", str(self.sample_shift),
                "--sample_guide_scale", str(self.guide_scale),
                "--base_seed", str(self.base_seed)
            ]
            
            logger.info(f"Executing Wan command: {' '.join(command)}")
            
            # If distributed training is enabled in config, modify the command
            if self.config.use_distributed:
                # Create a temporary shell script to run with torchrun
                script_path = os.path.join(self.temp_dir, f"run_wan_{unique_id}.sh")
                with open(script_path, "w") as f:
                    torchrun_cmd = f"torchrun --nproc_per_node={self.config.gpus_per_node} --master_port={random.randint(20000, 30000)} {self.config.wan_script_path}"
                    f.write("#!/bin/bash\n")
                    f.write(f"{torchrun_cmd} \\\n")
                    f.write(f"  --dit_fsdp --t5_fsdp \\\n")
                    f.write(f"  --task {self.task} \\\n")
                    f.write(f"  --size {self.size} \\\n")
                    f.write(f"  --ckpt_dir {self.model_path} \\\n")
                    f.write(f"  --save_file {output_video} \\\n")
                    f.write(f"  --first_frame {first_frame_path} \\\n")
                    f.write(f"  --last_frame {last_frame_path} \\\n")
                    f.write(f"  --prompt \"{transition_prompt}\" \\\n")
                    f.write(f"  --sample_steps {self.sample_steps} \\\n")
                    f.write(f"  --sample_shift {self.sample_shift} \\\n") 
                    f.write(f"  --sample_guide_scale {self.guide_scale} \\\n")
                    f.write(f"  --base_seed {self.base_seed}\n")
                
                # Make the script executable
                os.chmod(script_path, 0o755)
                
                # Execute the script
                logger.info(f"Executing distributed Wan script: {script_path}")
                result = subprocess.run([script_path], capture_output=True, text=True)
            else:
                # Execute the regular command
                result = subprocess.run(command, capture_output=True, text=True)
            
            # Check if generation was successful
            if not os.path.exists(output_video):
                logger.error(f"Wan video generation failed: {result.stderr}")
                raise Exception(f"Failed to generate video transition: {result.stderr}")
            
            logger.info(f"Successfully generated video transition: {output_video}")
            return output_video
            
        except Exception as e:
            logger.error(f"Error in Wan video generation: {str(e)}")
            raise
    
    def generate_transition_from_script(self, first_frame, last_frame, transition_prompt):
        """Generate a video transition by executing the Wan bash script.
        
        Args:
            first_frame: PIL Image of the first keyframe
            last_frame: PIL Image of the last keyframe
            transition_prompt: Text description of the transition
            
        Returns:
            str: Path to the generated video file
        """
        logger.info("Generating video transition using Wan bash script...")
        
        try:
            # Save keyframes to temporary directory
            first_frame_path = self._save_image_for_wan(first_frame, "first_frame.png")
            last_frame_path = self._save_image_for_wan(last_frame, "last_frame.png")
            
            # Create a unique ID for this generation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = f"{timestamp}_{random.randint(1000, 9999)}"
            
            # Define output video path
            output_video = os.path.join(self.temp_dir, f"transition_{unique_id}.mp4")
            
            # Create a temporary bash script
            script_path = os.path.join(self.temp_dir, f"run_wan_{unique_id}.sh")
            with open(script_path, "w") as f:
                f.write("#!/bin/bash\n")
                
                # Add environment setup if needed
                if hasattr(self.config, 'pip_requirements'):
                    for req in self.config.pip_requirements:
                        f.write(f"pip install {req} -i https://pypi.tuna.tsinghua.edu.cn/simple\n")
                
                # Add the Wan execution command
                f.write("# Set general parameters\n")
                f.write(f"TASK=\"{self.task}\"\n")
                f.write(f"SIZE=\"{self.size}\"\n")
                f.write(f"CKPT_DIR=\"{self.model_path}\"\n")
                f.write(f"SAVE_DIR=\"{self.temp_dir}\"\n\n")
                
                f.write("# Ensure save directory exists\n")
                f.write("mkdir -p $SAVE_DIR\n\n")
                
                f.write("echo \"Starting video transition generation...\"\n")
                f.write(f"torchrun --nproc_per_node={self.config.gpus_per_node} --master_port={random.randint(20000, 30000)} {self.config.wan_script_path} --dit_fsdp --t5_fsdp \\\n")
                f.write(f"    --task $TASK \\\n")
                f.write(f"    --size $SIZE \\\n")
                f.write(f"    --ckpt_dir $CKPT_DIR \\\n")
                f.write(f"    --save_file {output_video} \\\n")
                f.write(f"    --first_frame {first_frame_path} \\\n")
                f.write(f"    --last_frame {last_frame_path} \\\n")
                f.write(f"    --prompt \"{transition_prompt}\"\n\n")
                
                f.write("echo \"Video transition generation completed!\"\n")
            
            # Make the script executable
            os.chmod(script_path, 0o755)
            
            # Execute the script
            logger.info(f"Executing Wan bash script: {script_path}")
            result = subprocess.run([script_path], capture_output=True, text=True)
            
            # Verify successful generation
            if not os.path.exists(output_video):
                logger.error(f"Wan video generation failed: {result.stderr}")
                raise Exception(f"Failed to generate video transition: {result.stderr}")
            
            logger.info(f"Successfully generated video transition: {output_video}")
            return output_video
            
        except Exception as e:
            logger.error(f"Error in Wan video generation: {str(e)}")
            raise

    def __del__(self):
        """Clean up temporary files when the object is destroyed."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            logger.info(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)