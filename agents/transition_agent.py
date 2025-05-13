#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Transition Agent responsible for creating transitions between keyframes.
"""

import logging
from models.qwen_connector import QwenModel
from utils.prompt_templates import get_transition_prompt
import numpy as np
from PIL import Image
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TransitionAgent:
    """Agent responsible for generating transition prompts between keyframes."""
    
    def __init__(self, config):
        """Initialize the Transition Agent with configuration settings.
        
        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        logger.info("Initializing Qwen model for Transition Agent...")
        self.model = QwenModel(config)
        logger.info("Qwen model successfully initialized for Transition Agent")
        
    def create_transition_prompt(self, source_image, target_image, source_prompt, target_prompt, product_context=""):
        """Generate a transition prompt between two keyframes.
        
        Args:
            source_image: PIL Image of the starting keyframe
            target_image: PIL Image of the ending keyframe
            source_prompt: Text description of the starting keyframe
            target_prompt: Text description of the ending keyframe
            product_context: Additional context about the product
            
        Returns:
            str: Transition prompt
        """
        logger.info("Creating transition prompt between keyframes...")
        
        # Verify that images are in PIL Image format
        if not isinstance(source_image, Image.Image) or not isinstance(target_image, Image.Image):
            logger.error(f"Invalid image types. source_image: {type(source_image)}, target_image: {type(target_image)}")
            return self._create_fallback_transition(source_prompt, target_prompt)
        
        # Construct prompt for the transition model
        base_prompt = get_transition_prompt(source_prompt, target_prompt)
        
        # Incorporate product context if provided
        if product_context:
            base_prompt = f"{product_context}\n\n{base_prompt}"
        
        # Ensure consistent prompt formatting
        formatted_prompt = (
            f"I need a detailed description for a smooth transition between these two frames "
            f"for a professional marketing video.\n\n{base_prompt}\n\n"
            f"Create a specific, detailed description of how to transition from frame 1 to frame 2, "
            f"focusing on camera movement, visual effects, and timing."
        )
        
        # Prepare image inputs
        try:
            # Create copies of images to avoid modifying originals
            source_img = source_image.copy()
            target_img = target_image.copy()
            
            # Create image description objects
            images = [
                {"data": source_img, "description": "Source keyframe"},
                {"data": target_img, "description": "Target keyframe"}
            ]
            
            # Set maximum attempts for model call
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    # Call Qwen model
                    logger.info(f"Generating transition description (attempt {attempt+1}/{max_attempts})...")
                    response = self.model.generate(
                        system_prompt=self.config.transition_system_prompt,
                        user_prompt=formatted_prompt,
                        images=images
                    )
                    
                    # Check for common error patterns in response
                    if "I apologize" in response or "technical issue" in response:
                        logger.warning(f"Received error response on attempt {attempt+1}. Retrying...")
                        continue
                        
                    logger.info("Transition prompt generated successfully")
                    return response.strip()
                    
                except Exception as e:
                    logger.error(f"Error generating transition on attempt {attempt+1}: {str(e)}")
                    
                    # Create fallback prompt on final attempt
                    if attempt == max_attempts - 1:
                        logger.warning("All attempts failed. Creating fallback transition prompt.")
                        return self._create_fallback_transition(source_prompt, target_prompt)
        except Exception as e:
            logger.error(f"Error preparing images for transition: {str(e)}")
            return self._create_fallback_transition(source_prompt, target_prompt)
        
        # Safety fallback (should never be reached)
        return self._create_fallback_transition(source_prompt, target_prompt)

    def _create_fallback_transition(self, source_prompt, target_prompt):
        """Create a fallback transition prompt when the model call fails.
        
        Args:
            source_prompt: Description of source keyframe
            target_prompt: Description of target keyframe
            
        Returns:
            str: Fallback transition prompt
        """
        # Extract basic descriptions
        source_desc = source_prompt.split(":")[1].strip() if ":" in source_prompt else "first scene"
        target_desc = target_prompt.split(":")[1].strip() if ":" in target_prompt else "second scene"
        
        # Create a generic but effective transition prompt
        transitions = [
            f"Smooth dissolve transition from the {source_desc} to the {target_desc}, with gradual fading between elements.",
            f"Elegant zoom transition that slowly pushes in on the first frame and pulls out to reveal the second frame.",
            f"Cinematic wipe effect moving from left to right, transitioning from the {source_desc} to the {target_desc}.",
            f"Cross-fade between frames with a subtle blur effect, gradually revealing the new scene's details.",
            f"Professional slide transition that moves the first scene out while bringing in the second scene."
        ]
        
        return np.random.choice(transitions)