#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Connector for OmniControl model using FLUX pipeline.
"""

import os
import torch
from PIL import Image
import io
import base64
import logging
import traceback
from diffusers.pipelines import FluxPipeline
from src.flux.condition import Condition
from src.flux.generate import generate, seed_everything
# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class OmniControlModel:
    """Connector for the OmniControl model using FLUX pipeline."""
    
    def __init__(self, config):
        """Initialize the OmniControl model connector.
        
        Args:
            config: Configuration containing model paths and parameters
        """
        self.config = config
        self._load_model()
    
    def _load_model(self):
        """Load the FLUX pipeline with OmniControl LoRA."""
        # Load the base FLUX pipeline
        self.pipe = FluxPipeline.from_pretrained(
            self.config.flux_model_path, 
            torch_dtype=torch.bfloat16
        )
        self.pipe = self.pipe.to("cuda")
        
        # Load the OmniControl LoRA weights
        self.pipe.load_lora_weights(
            self.config.omnicontrol_path,
            weight_name=self.config.omnicontrol_weight_name,
            adapter_name="subject"
        )
    
    def _decode_base64_to_image(self, base64_string):
        """Decode base64 string to PIL Image.
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            PIL.Image: Decoded image
        """
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    
    def _encode_image_to_base64(self, image):
        """Encode PIL Image to base64 string.
        
        Args:
            image: PIL Image
            
        Returns:
            str: Base64 encoded image string
        """
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def generate(self, prompt, reference_image=None, seed=None, reference_weight=1.0):
        """Generate an image using the OmniControl model."""
        try:
            # Set random seed if provided
            if seed is not None:
                seed_everything(seed)
            else:
                seed_everything(self.config.default_seed)
            
            # Default image size
            height = self.config.image_size[0]
            width = self.config.image_size[1]
            
            # Add detailed debug logging
            logger.info(f"OmniControl generating with prompt: {prompt[:50]}...")
            logger.info(f"Reference image provided: {reference_image is not None}")
            logger.info(f"Reference weight: {reference_weight}")
            
            if reference_image:
                # Add extra logging to confirm reference image processing
                logger.info("Processing reference image for OmniControl...")
                
                # Convert base64 reference image to PIL Image
                ref_img = self._decode_base64_to_image(reference_image)
                original_size = ref_img.size
                ref_img = ref_img.resize((height, width))
                logger.info(f"Reference image resized from {original_size} to {height}x{width}")
                
                # Adjust position_delta based on reference_weight
                adjusted_position_delta = (
                    self.config.position_delta[0] * reference_weight,
                    self.config.position_delta[1] * reference_weight
                )
                
                logger.info(f"Using adjusted position delta: {adjusted_position_delta}")
                
                try:
                    # Create condition for subject control
                    # Modified here - removed weight parameter since Condition class does not accept it
                    condition = Condition(
                        "subject", 
                        ref_img, 
                        # position_delta=adjusted_position_delta
                        position_delta=(0,32),  # Use default position offset
                    )
                    
                    logger.info("Created subject condition, generating image...")
                    
                    # Generate the image with the reference condition
                    result_img = generate(
                        self.pipe,
                        prompt=prompt,
                        conditions=[condition],
                        num_inference_steps=self.config.num_inference_steps,
                        # guidance_scale=self.config.guidance_scale,
                        height=height,
                        width=width,
                    )
                    
                    # Check if an image was returned
                    if not hasattr(result_img, 'images') or len(result_img.images) == 0:
                        raise Exception("No images returned from OmniControl generator")
                    
                    result_img = result_img.images[0]
                    
                    logger.info("Image generation with reference condition complete")
                    
                    # Create a combined image showing reference and result side by side
                    concat_image = Image.new("RGB", (width * 2, height))
                    concat_image.paste(ref_img, (0, 0))
                    concat_image.paste(result_img, (width, 0))
                    
                    logger.info("Created comparison image with reference and result")
                    
                    return {
                        "image": self._encode_image_to_base64(result_img),
                        "combined_image": self._encode_image_to_base64(concat_image)
                    }
                except Exception as e:
                    logger.error(f"Error using reference condition: {str(e)}")
                    logger.warning("Falling back to generation without reference")
                    # Continue with non-reference image generation
            
            # If no reference image is provided or reference image processing fails
            logger.warning("No reference image provided! Generating without reference.")
            
            # Generate without reference condition
            result = generate(
                self.pipe,
                prompt=prompt,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                height=height,
                width=width,
            )
            
            # Check the result
            if not hasattr(result, 'images') or len(result.images) == 0:
                raise Exception("No images returned from OmniControl generator")
                
            result_img = result.images[0]
            
            return {
                "image": self._encode_image_to_base64(result_img)
            }
        except Exception as e:
            logger.error(f"Error in OmniControl generate: {str(e)}")
            # Provide more diagnostic information in case of errors
            if reference_image:
                logger.error(f"Reference image base64 length: {len(reference_image)}")
            logger.error(f"Prompt: {prompt}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise