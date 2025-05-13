#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Director Agent responsible for creating the narrative and keyframe prompts.
"""

import logging
from models.qwen_connector import QwenModel
from utils.prompt_templates import get_director_prompt
from PIL import Image
import re
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DirectorAgent:
    """Agent that generates keyframe descriptions and narrative for the advertisement."""
    
    def __init__(self, config):
        """Initialize the director agent.
        
        Args:
            config: Configuration object containing model parameters
        """
        self.config = config
        logger.info("Initializing Qwen model for director agent...")
        self.model = QwenModel(config)
        logger.info("Qwen model initialized for director agent")
    
    def create_keyframe_prompts(self, product_image, num_keyframes=5, additional_context=""):
        """Generate prompts for keyframes based on the product image.
        
        Args:
            product_image: PIL Image of the product
            num_keyframes: Number of keyframes to generate
            additional_context: Additional context about the product to guide generation
            
        Returns:
            list: List of keyframe prompts
        """
        logger.info(f"Creating keyframe prompts for {num_keyframes} keyframes...")
        
        # Ensure product_image is a PIL Image
        if not isinstance(product_image, Image.Image):
            logger.error(f"Invalid product_image type: {type(product_image)}")
            # Create a default response
            default_prompts = [
                f"Scene {i+1}: This item shown in an attractive setting with professional lighting."
                for i in range(num_keyframes)
            ]
            return default_prompts
        
        # Ensure a copy of the image is used, not the original
        image_copy = product_image.copy()
        
        # Build the prompt
        base_prompt = get_director_prompt(num_keyframes)
        
        # Add additional context
        if additional_context:
            prompt = f"I need you to create compelling keyframe descriptions for a marketing video for this item in the image.\n\nProduct Context: {additional_context}\n\n{base_prompt}"
        else:
            prompt = f"I need you to create compelling keyframe descriptions for a marketing video for this item in the image.\n\n{base_prompt}"
        
        # Set maximum number of retries
        max_retries = 3
        best_response = None
        best_quality_score = 0
        
        for attempt in range(max_retries):
            logger.info(f"Generating keyframe descriptions (attempt {attempt+1}/{max_retries})...")
            
            try:
                # Call the Qwen model
                response = self.model.generate(
                    system_prompt=self.config.director_system_prompt,
                    user_prompt=prompt,
                    image=image_copy  # Use the copy
                )
                
                # Evaluate response quality
                quality_score = self._evaluate_keyframe_quality(response, num_keyframes)
                logger.info(f"Response quality score: {quality_score}")
                
                if quality_score > best_quality_score:
                    best_response = response
                    best_quality_score = quality_score
                    
                    # If a high-quality response is obtained, no need to retry
                    if quality_score > 0.8:
                        logger.info("Found high-quality response, stopping retries.")
                        break
            except Exception as e:
                logger.error(f"Error in keyframe description generation (attempt {attempt+1}): {str(e)}")
        
        # If all attempts fail
        if best_response is None:
            logger.warning("All generation attempts failed. Using default keyframe prompts.")
            default_prompts = [
                f"Scene {i+1}: This item shown in an attractive setting with professional lighting."
                for i in range(num_keyframes)
            ]
            return default_prompts
        
        logger.info("Parsing keyframe descriptions...")
        # Parse the best response to extract keyframe prompts
        keyframe_prompts = self._parse_keyframe_response(best_response, num_keyframes)
        
        # Validate and fix keyframe prompts
        validated_prompts = self._validate_keyframe_prompts(keyframe_prompts, num_keyframes, additional_context)
        
        logger.info(f"Generated {len(validated_prompts)} keyframe prompts")
        return validated_prompts

    def _evaluate_keyframe_quality(self, response, num_keyframes):
        """Evaluate the quality of keyframe descriptions.
        
        Args:
            response: Model response containing keyframe descriptions
            num_keyframes: Expected number of keyframes
            
        Returns:
            float: Quality score between 0 and 1
        """
        # Check if we have enough keyframes
        keyframes = self._parse_keyframe_response(response, num_keyframes)
        if len(keyframes) < num_keyframes:
            return 0.3  # Penalty for not generating enough keyframes
        
        # Check for diversity in descriptions
        unique_words = set()
        total_words = 0
        repeated_phrases = 0
        
        for frame in keyframes:
            words = frame.lower().split()
            total_words += len(words)
            unique_words.update(words)
            
            # Check for repeated generic phrases
            generic_phrases = [
                "product shot with professional lighting",
                "professional lighting and attractive background",
                "professional studio lighting"
            ]
            for phrase in generic_phrases:
                if phrase in frame.lower():
                    repeated_phrases += 1
        
        # Unique word ratio (higher is better)
        word_diversity = len(unique_words) / total_words if total_words > 0 else 0
        
        # Penalize for repeated generic phrases
        repetition_penalty = max(0, 1.0 - (repeated_phrases / num_keyframes))
        
        # Check for specificity and detail
        detail_score = 0
        for frame in keyframes:
            # Points for specific camera directions
            if any(term in frame.lower() for term in ["close-up", "wide shot", "angle", "zoom", "pan"]):
                detail_score += 0.5
                
            # Points for lighting specifics
            if any(term in frame.lower() for term in ["lighting", "shadow", "highlight", "contrast"]):
                detail_score += 0.5
                
            # Points for environment details
            if any(term in frame.lower() for term in ["background", "setting", "environment", "scene"]):
                detail_score += 0.5
        
        # Normalize detail score
        detail_score = min(1.0, detail_score / (num_keyframes * 2))
        
        # Combine scores with weights
        final_score = (word_diversity * 0.3) + (repetition_penalty * 0.4) + (detail_score * 0.3)
        return final_score

    def _evaluate_keyframe_quality(self, response, num_keyframes):
        """Evaluate the quality of keyframe descriptions.
        
        Args:
            response: Model response containing keyframe descriptions
            num_keyframes: Expected number of keyframes
            
        Returns:
            float: Quality score between 0 and 1
        """
        # Check if we have enough keyframes
        keyframes = self._parse_keyframe_response(response, num_keyframes)
        if len(keyframes) < num_keyframes:
            return 0.3  # Penalty for not generating enough keyframes
        
        # Check for diversity in descriptions
        unique_words = set()
        total_words = 0
        repeated_phrases = 0
        
        for frame in keyframes:
            words = frame.lower().split()
            total_words += len(words)
            unique_words.update(words)
            
            # Check for repeated generic phrases
            generic_phrases = [
                "product shot with professional lighting",
                "professional lighting and attractive background",
                "professional studio lighting"
            ]
            for phrase in generic_phrases:
                if phrase in frame.lower():
                    repeated_phrases += 1
        
        # Unique word ratio (higher is better)
        word_diversity = len(unique_words) / total_words if total_words > 0 else 0
        
        # Penalize for repeated generic phrases
        repetition_penalty = max(0, 1.0 - (repeated_phrases / num_keyframes))
        
        # Check for specificity and detail
        detail_score = 0
        for frame in keyframes:
            # Points for specific camera directions
            if any(term in frame.lower() for term in ["close-up", "wide shot", "angle", "zoom", "pan"]):
                detail_score += 0.5
                
            # Points for lighting specifics
            if any(term in frame.lower() for term in ["lighting", "shadow", "highlight", "contrast"]):
                detail_score += 0.5
                
            # Points for environment details
            if any(term in frame.lower() for term in ["background", "setting", "environment", "scene"]):
                detail_score += 0.5
        
        # Normalize detail score
        detail_score = min(1.0, detail_score / (num_keyframes * 2))
        
        # Combine scores with weights
        final_score = (word_diversity * 0.3) + (repetition_penalty * 0.4) + (detail_score * 0.3)
        return final_score

    def _validate_keyframe_prompts(self, keyframe_prompts, num_keyframes, product_context):
        """Validate and fix keyframe prompts to ensure diversity and quality.
        
        Args:
            keyframe_prompts: List of generated keyframe prompts
            num_keyframes: Expected number of keyframes
            product_context: Product context information
            
        Returns:
            list: List of validated and fixed keyframe prompts
        """
        validated_prompts = []
        
        # Extract product type from context
        product_type = "product"
        if "product is a" in product_context:
            product_type = product_context.split("product is a")[1].split(".")[0].strip()
        
        # Define diverse creative templates for fallbacks
        diverse_templates = [
            f"This item floats on calm water reflecting a colorful sunset.",
            f"Under a spotlight in a dark room, this item casts dramatic shadows.",
            f"This item sits on an old wooden table with morning light streaming through a window.",
            f"Nestled among vibrant flowers in a garden, this item draws the eye.",
            f"This item appears balanced on the edge of a skyscraper with the city below."
        ]
        
        # Track words to ensure diversity
        common_words = set()
        generic_count = 0
        
        for i, prompt in enumerate(keyframe_prompts):
            # Check if this is a generic/default prompt
            is_generic = "professional lighting" in prompt.lower() or "attractive background" in prompt.lower()
            
            # Check word overlap with previous prompts
            prompt_words = set(w.lower() for w in prompt.split() if len(w) > 4)
            overlap = prompt_words.intersection(common_words)
            high_overlap = len(overlap) > 3 and len(overlap) / len(prompt_words) > 0.5
            
            if is_generic or high_overlap:
                generic_count += 1
                if generic_count > 1:
                    # Use a creative template instead
                    template_idx = i % len(diverse_templates)
                    new_prompt = diverse_templates[template_idx]
                    validated_prompts.append(new_prompt)
                else:
                    # First generic prompt, keep it but mark as seen
                    validated_prompts.append(prompt)
            else:
                # Not generic, use as is
                validated_prompts.append(prompt)
                # Add words to common set
                common_words.update(prompt_words)
        
        # If we don't have enough prompts, add more using templates
        while len(validated_prompts) < num_keyframes:
            i = len(validated_prompts)
            template_idx = i % len(diverse_templates)
            new_prompt = diverse_templates[template_idx]
            validated_prompts.append(new_prompt)
        
        return validated_prompts
    
    def _parse_keyframe_response(self, response, num_keyframes):
        """Parse the response from the model to extract keyframe prompts.
        
        Args:
            response: Response from the model
            num_keyframes: Expected number of keyframes
            
        Returns:
            list: List of keyframe prompts
        """
        keyframe_prompts = []
        
        # Try to extract numbered keyframes/scenes
        pattern = r"(?:KEYFRAME|SCENE|)\s*(\d+)[:\.\)]+\s*(.*?)(?=(?:KEYFRAME|SCENE|)\s*\d+[:\.\)]|$)"
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            # Process numbered matches
            for idx, desc in matches:
                # Clean up and add the prompt
                clean_desc = desc.strip()
                if clean_desc:
                    keyframe_prompts.append(clean_desc)
        else:
            # If no numbered format, try paragraph splitting
            paragraphs = [p.strip() for p in response.split("\n\n") if p.strip()]
            for p in paragraphs:
                if len(p) > 10 and not p.startswith("I apologize") and not p.startswith("Here are"):
                    keyframe_prompts.append(p)
        
        # Ensure we have at least one keyframe
        if not keyframe_prompts:
            logger.warning("Could not parse any keyframe prompts from response. Using generic prompt.")
            keyframe_prompts = ["This item displayed in an attractive setting with professional lighting."]
        
        # Ensure we have the expected number of keyframes
        while len(keyframe_prompts) < num_keyframes:
            idx = len(keyframe_prompts) + 1
            keyframe_prompts.append(f"Scene {idx}: This item shown from a different angle in an interesting environment.")
        
        # Trim to the expected number
        if len(keyframe_prompts) > num_keyframes:
            keyframe_prompts = keyframe_prompts[:num_keyframes]
        
        return keyframe_prompts