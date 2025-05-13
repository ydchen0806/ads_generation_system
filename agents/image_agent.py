#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image Agent responsible for generating keyframe images based on prompts.
"""

import base64
from io import BytesIO
from PIL import Image
import os
import logging
import numpy as np
from models.omnicontrol_connector import OmniControlModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageAgent:
    """Agent that generates keyframe images based on text prompts using OmniControl."""
    
    def __init__(self, config):
        """Initialize the Image Agent with configuration settings.
        
        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        logger.info("Initializing OmniControl model...")
        self.model = OmniControlModel(config)
        logger.info("OmniControl model successfully initialized")
    
    def _encode_image(self, image):
        """Encode an image to base64 for model input.
        
        Args:
            image: PIL Image object
            
        Returns:
            str: Base64 encoded image string
        """
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    def _decode_image(self, base64_string):
        """Decode a base64 string to a PIL Image.
        
        Args:
            base64_string: Base64 encoded image string
            
        Returns:
            PIL.Image: Decoded image
        """
        image_data = base64.b64decode(base64_string)
        return Image.open(BytesIO(image_data))
    
    def _enhance_prompt(self, prompt):
        """Enhance the prompt with additional details for improved image generation.
        
        Args:
            prompt: Original prompt
            
        Returns:
            str: Enhanced prompt
        """
        # Add detailed descriptors and style preferences to improve generation quality
        enhanced_prompt = f"{prompt}, professional photography, detailed, high quality, 4K, sharp focus"
        
        # Include negative prompt elements if specified in config
        if hasattr(self.config, 'negative_prompt'):
            enhanced_prompt += f" ||| {self.config.negative_prompt}"
            
        return enhanced_prompt
    
    def _prepare_reference_image(self, reference_image):
        """Prepare a reference image for OmniControl by ensuring proper size and format.
        
        Args:
            reference_image: Original PIL Image
            
        Returns:
            PIL.Image: Processed reference image
        """
        # Resize image to the model's expected dimensions
        target_size = self.config.image_size
        processed_image = reference_image.resize(target_size, Image.LANCZOS)
        
        # Ensure image is in RGB format
        if processed_image.mode != "RGB":
            processed_image = processed_image.convert("RGB")
            
        return processed_image
    
    def generate(self, prompt, seed=None):
        """Generate an image based solely on a text prompt.
        
        Args:
            prompt: Text prompt describing the desired image
            seed: Optional seed for reproducibility
            
        Returns:
            PIL.Image: Generated image
        """
        logger.info(f"Generating image for prompt: {prompt[:50]}...")
        
        # Enhance the prompt for better results
        enhanced_prompt = self._enhance_prompt(prompt)
        
        # Generate image using the model
        try:
            response = self.model.generate(prompt=enhanced_prompt, seed=seed)
            logger.info("Image generation successful")
            
            # Decode and return the image
            return self._decode_image(response["image"])
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            # Fallback to a default image if available
            if hasattr(self.config, 'fallback_image_path') and os.path.exists(self.config.fallback_image_path):
                logger.info("Using fallback image")
                return Image.open(self.config.fallback_image_path)
            else:
                # Create a blank image with error text as fallback
                img = Image.new('RGB', self.config.image_size, color=(255, 255, 255))
                logger.info("Created blank image as fallback")
                return img
    
    def generate_with_reference(self, prompt, reference_image, seed=None, reference_weight=1.0):
        """Generate an image based on a text prompt and a reference image.
        
        Args:
            prompt: Text prompt describing the desired image
            reference_image: PIL Image to use as reference
            seed: Optional seed for reproducibility
            reference_weight: Weight to apply to the reference image (0.0-1.0)
            
        Returns:
            PIL.Image: Generated image
        """
        logger.info(f"Generating image with reference (weight={reference_weight}) for prompt: {prompt[:50]}...")
        
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                # Log reference image details
                logger.info(f"Reference image size: {reference_image.size}, mode: {reference_image.mode}")
                
                # Enhance the prompt for better results
                enhanced_prompt = self._enhance_prompt(prompt)
                
                # Prepare reference image
                processed_ref = self._prepare_reference_image(reference_image)
                encoded_reference = self._encode_image(processed_ref)
                
                # Generate image
                response = self.model.generate(
                    prompt=enhanced_prompt,
                    reference_image=encoded_reference,
                    seed=seed,
                    reference_weight=reference_weight
                )
                
                # Decode and return the generated image
                return self._decode_image(response["image"])
                
            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {str(e)}")
                if attempt == max_attempts - 1:
                    logger.error(f"All {max_attempts} attempts to generate with reference failed")
                    raise
        
        # This code should never be reached, as the loop will raise an exception on failure
        raise Exception("Unexpected code path in generate_with_reference")