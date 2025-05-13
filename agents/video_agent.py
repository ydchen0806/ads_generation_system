#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video Agent responsible for generating video transitions and assembling the final video.
"""

import os
import logging
import tempfile
import shutil
import subprocess
from PIL import Image
from moviepy.editor import VideoFileClip, ImageClip, concatenate_videoclips
from models.wan_connector import WanModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoAgent:
    """Agent that generates video transitions and assembles the final marketing video."""
    
    def __init__(self, config):
        """Initialize the video agent.
        
        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        logger.info("Initializing Wan model for video generation...")
        self.model = WanModel(config)
        logger.info("Wan model initialized")
        
        # Create temporary directory for working files
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary work directory: {self.temp_dir}")
    
    def generate_transition(self, source_image, target_image, transition_prompt, use_script=True):
        """Generate a video transition between two keyframes.
        
        Args:
            source_image: PIL Image of the starting keyframe
            target_image: PIL Image of the ending keyframe
            transition_prompt: Text description of the transition
            use_script: Whether to use the bash script method (True) or direct Python call (False)
            
        Returns:
            VideoClip: Generated transition video
        """
        logger.info(f"Generating transition between keyframes...")
        
        try:
            # Process keyframes if needed (resize, enhance, etc)
            source_image = self._prepare_keyframe(source_image)
            target_image = self._prepare_keyframe(target_image)
            
            # Generate the transition video
            if use_script:
                video_path = self.model.generate_transition_from_script(
                    source_image, target_image, transition_prompt
                )
            else:
                video_path = self.model.generate_transition(
                    source_image, target_image, transition_prompt
                )
            
            # Create VideoClip object
            logger.info(f"Loading generated transition video: {video_path}")
            return VideoClip(video_path)
            
        except Exception as e:
            logger.error(f"Error generating transition: {str(e)}")
            # Fallback to a simple fade transition
            return self._create_fallback_transition(source_image, target_image)
    
    def _prepare_keyframe(self, image):
        """Prepare a keyframe for video generation.
        
        Args:
            image: PIL Image to prepare
            
        Returns:
            PIL.Image: Processed image
        """
        # Ensure image is in RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize to match the expected Wan input resolution
        wan_size = [int(dim) for dim in self.model.size.split("*")]
        if image.size != tuple(wan_size):
            # Preserve aspect ratio by resizing and center cropping
            aspect_ratio = wan_size[0] / wan_size[1]
            img_aspect = image.width / image.height
            
            if img_aspect > aspect_ratio:
                # Image is wider than target
                new_height = image.height
                new_width = int(new_height * aspect_ratio)
                left = (image.width - new_width) // 2
                image = image.crop((left, 0, left + new_width, new_height))
            else:
                # Image is taller than target
                new_width = image.width
                new_height = int(new_width / aspect_ratio)
                top = (image.height - new_height) // 2
                image = image.crop((0, top, new_width, top + new_height))
            
            # Resize to target dimensions
            image = image.resize(wan_size, Image.LANCZOS)
        
        return image
    
    def _create_fallback_transition(self, source_image, target_image):
        """Create a simple fallback transition in case the Wan model fails.
        
        Args:
            source_image: First keyframe PIL Image
            target_image: Last keyframe PIL Image
            
        Returns:
            VideoClip: Fallback transition video
        """
        logger.warning("Creating fallback transition using ffmpeg...")
        
        try:
            # Save frames to temporary files
            source_path = os.path.join(self.temp_dir, "fallback_source.png")
            target_path = os.path.join(self.temp_dir, "fallback_target.png")
            source_image.save(source_path)
            target_image.save(target_path)
            
            # Create output path
            output_path = os.path.join(self.temp_dir, "fallback_transition.mp4")
            
            # Use ffmpeg to create a crossfade
            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-t", "1.5", "-i", source_path,
                "-loop", "1", "-t", "1.5", "-i", target_path,
                "-filter_complex", 
                "[0:v]fade=t=out:st=1:d=0.5[v0];[1:v]fade=t=in:st=0:d=0.5[v1];[v0][v1]overlay=shortest=1",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", output_path
            ]
            
            subprocess.run(cmd, check=True)
            
            if os.path.exists(output_path):
                logger.info("Successfully created fallback transition")
                return VideoClip(output_path)
            else:
                raise Exception("Failed to create fallback transition")
            
        except Exception as e:
            logger.error(f"Error creating fallback transition: {str(e)}")
            # Return an even simpler transition (just an empty clip)
            return VideoClip(None, duration=1.0)
    
    def assemble_final_video(self, keyframe_images, transition_clips):
        """Assemble the final marketing video from keyframes and transitions.
        
        Args:
            keyframe_images: List of PIL Image keyframes
            transition_clips: List of VideoClip transitions
            
        Returns:
            VideoClip: Final assembled marketing video
        """
        logger.info("Assembling final marketing video...")
        
        try:
            # Create a temporary directory for assembly
            assembly_dir = os.path.join(self.temp_dir, "assembly")
            os.makedirs(assembly_dir, exist_ok=True)
            
            # Create a list to store all clip segments
            all_clips = []
            
            # Process each keyframe and transition
            for i, keyframe in enumerate(keyframe_images):
                # Add keyframe as a still image (3 seconds)
                keyframe_path = os.path.join(assembly_dir, f"keyframe_{i}.png")
                keyframe.save(keyframe_path)
                
                keyframe_clip = ImageClip(keyframe_path).set_duration(3)
                all_clips.append(keyframe_clip)
                
                # Add transition if there is one
                if i < len(transition_clips):
                    transition_path = transition_clips[i].path
                    if transition_path and os.path.exists(transition_path):
                        transition_clip = VideoFileClip(transition_path)
                        all_clips.append(transition_clip)
            
            # Concatenate all clips
            final_clip = concatenate_videoclips(all_clips)
            
            # Create output file path
            output_path = os.path.join(self.temp_dir, "final_marketing_video.mp4")
            
            # Write the final video
            logger.info(f"Writing final marketing video to {output_path}...")
            final_clip.write_videofile(
                output_path, 
                codec="libx264", 
                audio_codec="aac",
                fps=self.config.video_fps
            )
            
            logger.info("Final video assembly complete")
            return VideoClip(output_path)
            
        except Exception as e:
            logger.error(f"Error assembling final video: {str(e)}")
            raise
    
    def __del__(self):
        """Clean up when the object is destroyed."""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            logger.info(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)


class VideoClip:
    """Simple wrapper for video file paths with additional utility methods."""
    
    def __init__(self, path, duration=None):
        """Initialize a video clip.
        
        Args:
            path: Path to the video file, or None for a blank clip
            duration: Optional duration for blank clips
        """
        self.path = path
        self.duration = duration
        
        # If path is provided, get information about the video
        if path and os.path.exists(path):
            try:
                clip = VideoFileClip(path)
                self.duration = clip.duration
                self.fps = clip.fps
                self.size = clip.size
                clip.close()
            except Exception as e:
                logger.error(f"Error loading video clip {path}: {str(e)}")
                self.duration = duration or 3.0
                self.fps = 30
                self.size = (1280, 720)
    
    def save(self, output_path):
        """Save the video clip to a specified path.
        
        Args:
            output_path: Path to save the video to
        """
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if self.path and os.path.exists(self.path):
            # Copy the existing file
            shutil.copy2(self.path, output_path)
            logger.info(f"Saved video clip to {output_path}")
        else:
            # Create a blank video if there's no source
            logger.warning(f"No source video to save to {output_path}")
            blank_cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", f"color=c=black:s=1280x720:d={self.duration or 3.0}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
                output_path
            ]
            subprocess.run(blank_cmd, check=True)
            logger.info(f"Created blank video at {output_path}")