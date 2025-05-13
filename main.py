#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the advertisement generation system.
This script orchestrates the entire process from input image to marketing video,
and also provides development testing functionality for individual components.
"""

import os
import argparse
import logging
import re
import numpy as np
import time
from datetime import datetime
import shutil
import tempfile
from PIL import Image
try:
    from PIL import ImageDraw, ImageFont
    PIL_DRAW_AVAILABLE = True
except ImportError:
    PIL_DRAW_AVAILABLE = False
# Import configuration
from config import Config

# Import agents
from agents.director_agent import DirectorAgent
from agents.image_agent import ImageAgent
from agents.transition_agent import TransitionAgent
from agents.video_agent import VideoAgent

# Import models for direct testing
from models.qwen_connector import QwenModel
from models.omnicontrol_connector import OmniControlModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ad_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Advertisement Generation System')
    
    # Main subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Full advertisement generation pipeline
    pipeline_parser = subparsers.add_parser('pipeline', help='Run the full advertisement generation pipeline')
    pipeline_parser.add_argument('--image_path', type=str, required=True, help='Path to the product image')
    pipeline_parser.add_argument('--keyframes', type=int, default=5, help='Number of keyframes to generate')
    pipeline_parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the output')
    pipeline_parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    # Test image generation
    image_parser = subparsers.add_parser('image', help='Test image generation')
    image_parser.add_argument('--image_path', type=str, required=True, help='Path to the reference product image')
    image_parser.add_argument('--prompt', type=str, help='Test prompt for image generation')
    image_parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    # Test Qwen model
    qwen_parser = subparsers.add_parser('qwen', help='Test Qwen2.5-VL model')
    qwen_parser.add_argument('--image_path', type=str, required=True, help='Path to the image for description')
    qwen_parser.add_argument('--prompt', type=str, help='Custom prompt for the Qwen model')
    
    # Test director agent
    director_parser = subparsers.add_parser('director', help='Test director agent')
    director_parser.add_argument('--image_path', type=str, required=True, help='Path to the product image')
    director_parser.add_argument('--keyframes', type=int, default=5, help='Number of keyframes to generate')
    
    # Test transition agent
    transition_parser = subparsers.add_parser('transition', help='Test transition agent')
    transition_parser.add_argument('--source_image', type=str, required=True, help='Path to the source keyframe image')
    transition_parser.add_argument('--target_image', type=str, required=True, help='Path to the target keyframe image')
    transition_parser.add_argument('--source_prompt', type=str, help='Description of the source keyframe')
    transition_parser.add_argument('--target_prompt', type=str, help='Description of the target keyframe')
    
    # Test video agent
    video_parser = subparsers.add_parser('video', help='Test video agent')
    video_parser.add_argument('--keyframes_dir', type=str, required=True, help='Directory containing keyframe images')
    video_parser.add_argument('--transitions_dir', type=str, help='Directory containing transition prompts')
    video_parser.add_argument('--output_path', type=str, default='output/test_video.mp4', help='Output video path')
    
    return parser.parse_args()

def test_image_generation(config, image_path, prompt=None, seed=None):
    """Test the image generation component.
    
    Args:
        config: Configuration object
        image_path: Path to reference image
        prompt: Optional test prompt
        seed: Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing generated images
    """
    logger.info("Testing image generation...")
    
    # Initialize the image agent
    image_agent = ImageAgent(config)
    
    # Load the reference image
    reference_image = Image.open(image_path).convert("RGB")
    logger.info(f"Loaded reference image: {image_path}")
    
    # Use default prompt if none provided
    if prompt is None:
        prompt = "A professional marketing shot of this product on a sleek white surface with soft lighting"
    
    # Create output directory for this test
    test_output_dir = os.path.join(config.output_dir, f"image_test_{int(time.time())}")
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Save input data
    reference_image.save(os.path.join(test_output_dir, "reference.png"))
    with open(os.path.join(test_output_dir, "prompt.txt"), "w") as f:
        f.write(prompt)
    
    # Test image generation with reference
    logger.info("Testing generation with reference image...")
    result_with_ref = image_agent.generate_with_reference(prompt, reference_image, seed)
    result_with_ref.save(os.path.join(test_output_dir, "result_with_reference.png"))
    logger.info("Saved result with reference")
    
    # Test image generation without reference
    logger.info("Testing generation without reference image...")
    result_without_ref = image_agent.generate(prompt, seed)
    result_without_ref.save(os.path.join(test_output_dir, "result_without_reference.png"))
    logger.info("Saved result without reference")
    
    # Create a combined image for comparison
    combined = Image.new("RGB", (reference_image.width * 3, reference_image.height))
    combined.paste(reference_image, (0, 0))
    combined.paste(result_with_ref, (reference_image.width, 0))
    combined.paste(result_without_ref, (reference_image.width * 2, 0))
    combined.save(os.path.join(test_output_dir, "comparison.png"))
    logger.info("Saved comparison image")
    
    # Copy the comparison image to the main output directory for easy access
    shutil.copy(
        os.path.join(test_output_dir, "comparison.png"),
        os.path.join(config.output_dir, "latest_image_comparison.png")
    )
    
    logger.info(f"All results saved to {test_output_dir}")
    
    return {
        "reference": reference_image,
        "with_reference": result_with_ref,
        "without_reference": result_without_ref,
        "comparison": combined,
        "output_dir": test_output_dir
    }

def test_qwen_model(config, image_path, custom_prompt=None):
    """Test the Qwen2.5-VL model with an image.
    
    Args:
        config: Configuration object
        image_path: Path to test image
        custom_prompt: Optional custom prompt to use instead of the default
    
    Returns:
        str: Generated description
    """
    logger.info("Testing Qwen2.5-VL model...")
    
    # Initialize the Qwen model
    qwen_model = QwenModel(config)
    
    # Create test output directory
    test_output_dir = os.path.join(config.output_dir, f"qwen_test_{int(time.time())}")
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Copy the test image to the output directory
    shutil.copy(image_path, os.path.join(test_output_dir, "test_image.jpg"))
    
    # Use custom prompt or default
    if custom_prompt:
        test_prompt = custom_prompt
    else:
        test_prompt = "Describe this image in detail, including objects, colors, and mood."
    
    # Save the prompt
    with open(os.path.join(test_output_dir, "prompt.txt"), "w") as f:
        f.write(test_prompt)
    
    # Generate description
    logger.info("Generating image description...")
    response = qwen_model.generate(
        system_prompt="You are a helpful assistant that describes images accurately and in detail.",
        user_prompt=test_prompt,
        image=image_path
    )
    
    logger.info("Description generated:")
    logger.info(response)
    
    # Save the response to a text file
    output_path = os.path.join(test_output_dir, "description.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response)
    
    # Also save to main output directory for easy access
    with open(os.path.join(config.output_dir, "latest_qwen_description.txt"), "w", encoding="utf-8") as f:
        f.write(response)
        
    logger.info(f"Description saved to {output_path}")
    logger.info(f"All results saved to {test_output_dir}")
    
    return response

def test_director_agent(config, image_path, num_keyframes=5):
    """Test the director agent by generating keyframe prompts.
    
    Args:
        config: Configuration object
        image_path: Path to product image
        num_keyframes: Number of keyframes to generate
    
    Returns:
        list: Generated keyframe prompts
    """
    logger.info(f"Testing director agent with {num_keyframes} keyframes...")
    
    # Create test output directory
    test_output_dir = os.path.join(config.output_dir, f"director_test_{int(time.time())}")
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Copy the product image to the output directory
    shutil.copy(image_path, os.path.join(test_output_dir, "product_image.jpg"))
    
    # Initialize director agent
    director = DirectorAgent(config)
    
    # Load product image
    product_image = Image.open(image_path).convert("RGB")
    
    # Generate keyframe prompts
    logger.info("Generating keyframe prompts...")
    keyframe_prompts = director.create_keyframe_prompts(product_image, num_keyframes)
    
    # Save prompts to file
    output_path = os.path.join(test_output_dir, "keyframe_prompts.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(keyframe_prompts):
            f.write(f"KEYFRAME {i+1}:\n{prompt}\n\n")
    
    # Also save to main output directory for easy access
    with open(os.path.join(config.output_dir, "latest_keyframe_prompts.txt"), "w", encoding="utf-8") as f:
        for i, prompt in enumerate(keyframe_prompts):
            f.write(f"KEYFRAME {i+1}:\n{prompt}\n\n")
    
    logger.info(f"Keyframe prompts saved to {output_path}")
    logger.info(f"All results saved to {test_output_dir}")
    
    return keyframe_prompts

def test_transition_agent(config, source_image_path, target_image_path, source_prompt=None, target_prompt=None):
    """Test the transition agent by generating a transition prompt.
    
    Args:
        config: Configuration object
        source_image_path: Path to source keyframe image
        target_image_path: Path to target keyframe image
        source_prompt: Optional description of source keyframe
        target_prompt: Optional description of target keyframe
    
    Returns:
        str: Generated transition prompt
    """
    logger.info("Testing transition agent...")
    
    # Create test output directory
    test_output_dir = os.path.join(config.output_dir, f"transition_test_{int(time.time())}")
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Copy the input images to the output directory
    shutil.copy(source_image_path, os.path.join(test_output_dir, "source_keyframe.jpg"))
    shutil.copy(target_image_path, os.path.join(test_output_dir, "target_keyframe.jpg"))
    
    # Load keyframe images
    source_image = Image.open(source_image_path).convert("RGB")
    target_image = Image.open(target_image_path).convert("RGB")
    
    # Initialize transition agent
    transition_agent = TransitionAgent(config)
    
    # Use default prompts if none provided
    if source_prompt is None:
        source_prompt = "Keyframe 1: Product introduction on clean background"
    if target_prompt is None:
        target_prompt = "Keyframe 2: Product in use in lifestyle setting"
    
    # Save input prompts
    with open(os.path.join(test_output_dir, "input_prompts.txt"), "w", encoding="utf-8") as f:
        f.write(f"SOURCE PROMPT:\n{source_prompt}\n\nTARGET PROMPT:\n{target_prompt}")
    
    # Generate transition prompt
    logger.info("Generating transition prompt...")
    transition_prompt = transition_agent.create_transition_prompt(
        source_image, target_image, source_prompt, target_prompt
    )
    
    # Save transition prompt
    output_path = os.path.join(test_output_dir, "transition_prompt.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transition_prompt)
    
    # Also save to main output directory for easy access
    with open(os.path.join(config.output_dir, "latest_transition_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(transition_prompt)
    
    logger.info(f"Transition prompt saved to {output_path}")
    logger.info(f"All results saved to {test_output_dir}")
    
    return transition_prompt

def test_video_agent(config, keyframes_dir, transitions_dir=None, output_path=None):
    """Test the video agent by assembling keyframes into a video.
    
    Args:
        config: Configuration object
        keyframes_dir: Directory containing keyframe images
        transitions_dir: Optional directory containing transition prompts
        output_path: Optional path for the output video
    
    Returns:
        str: Path to the generated video
    """
    logger.info("Testing video agent...")
    
    # Create default output path if not provided
    if output_path is None:
        output_path = os.path.join(config.output_dir, f"test_video_{int(time.time())}.mp4")
    
    # Initialize video agent
    video_agent = VideoAgent(config)
    
    # Load keyframe images
    keyframe_paths = sorted([
        os.path.join(keyframes_dir, f) for f in os.listdir(keyframes_dir)
        if f.endswith(('.png', '.jpg', '.jpeg')) and 'keyframe' in f.lower()
    ])
    
    if not keyframe_paths:
        logger.error(f"No keyframe images found in {keyframes_dir}")
        return None
    
    keyframe_images = [Image.open(path).convert("RGB") for path in keyframe_paths]
    logger.info(f"Loaded {len(keyframe_images)} keyframe images")
    
    # Load transition prompts if provided
    transition_prompts = []
    if transitions_dir and os.path.exists(transitions_dir):
        transition_paths = sorted([
            os.path.join(transitions_dir, f) for f in os.listdir(transitions_dir)
            if f.endswith('.txt') and 'transition' in f.lower()
        ])
        
        for path in transition_paths:
            with open(path, 'r', encoding='utf-8') as f:
                transition_prompts.append(f.read().strip())
        
        logger.info(f"Loaded {len(transition_prompts)} transition prompts")
    
    # Generate transitions between keyframes
    logger.info("Generating transition videos...")
    transition_clips = []
    
    for i in range(len(keyframe_images) - 1):
        source_image = keyframe_images[i]
        target_image = keyframe_images[i+1]
        
        # Use loaded prompt if available, otherwise use a default
        if i < len(transition_prompts):
            transition_prompt = transition_prompts[i]
        else:
            transition_prompt = f"Smooth transition from keyframe {i+1} to keyframe {i+2}"
        
        logger.info(f"Generating transition {i+1} -> {i+2}...")
        transition_clip = video_agent.generate_transition(
            source_image, target_image, transition_prompt
        )
        transition_clips.append(transition_clip)
    
    # Assemble final video
    logger.info("Assembling final video...")
    final_video = video_agent.assemble_final_video(keyframe_images, transition_clips)
    final_video.save(output_path)
    
    logger.info(f"Final video saved to {output_path}")
    
    return output_path


def _extract_core_description(keyframe_prompt):
    """Extract the core description from a keyframe prompt.
    
    Args:
        keyframe_prompt: The full keyframe prompt text
        
    Returns:
        str: The extracted core description
    """
    # Default description in case extraction fails
    default_description = "Professional product shot with elegant composition."
    
    try:
        # Check if this is a fallback prompt
        if "Product shot with professional lighting" in keyframe_prompt:
            return default_description
        
        # Handle various formats of keyframe prompts
        if ":" in keyframe_prompt:
            # Extract the text after the first colon
            description = keyframe_prompt.split(":", 1)[1].strip()
            
            # If there are multiple paragraphs, take the first one
            if "\n\n" in description:
                description = description.split("\n\n")[0].strip()
                
            # Limit length to avoid overly long prompts
            if len(description) > 200:
                description = description[:200].strip() + "..."
                
            return description
        
        # Handle case where no pattern matches
        return default_description
    except Exception:
        return default_description

def _create_fallback_transition_prompt(source_prompt, target_prompt, product_info):
    """Create a fallback transition prompt when the main one fails.
    
    Args:
        source_prompt: Source keyframe prompt
        target_prompt: Target keyframe prompt
        product_info: Dictionary containing product information
        
    Returns:
        str: Fallback transition prompt
    """
    # Extract basic descriptions
    source_desc = _extract_core_description(source_prompt)
    target_desc = _extract_core_description(target_prompt)
    
    # Create a simple but effective transition prompt
    transition_type = np.random.choice([
        "dissolve", "zoom", "pan", "fade", "slide", "rotate"
    ])
    
    product_type = product_info.get("type", "product")
    
    fallback_prompt = (
        f"Create a smooth {transition_type} transition from the first frame showing {product_type} "
        f"({source_desc}) to the second frame ({target_desc}). "
        f"The transition should be elegant and professional, maintaining focus on the {product_type} "
        f"throughout. Use subtle lighting changes and smooth camera movement "
        f"to create a seamless flow between the scenes."
    )
    
    return fallback_prompt

def _extract_narrative_elements(keyframe_prompts):
    """Extract narrative elements from keyframe prompts for better transitions.
    
    Args:
        keyframe_prompts: List of keyframe prompts
        
    Returns:
        dict: Dictionary mapping keyframe index to narrative elements
    """
    narrative_elements = {}
    
    for i, prompt in enumerate(keyframe_prompts):
        elements = {
            "index": i,
            "title": "",
            "focus": "product",
            "mood": "professional",
            "environment": "studio",
            "lighting": "bright"
        }
        
        # Extract title
        if ":" in prompt:
            elements["title"] = prompt.split(":", 1)[0].strip()
            description = prompt.split(":", 1)[1].strip()
        else:
            description = prompt
        
        # Extract mood
        mood_keywords = {
            "dramatic": ["dramatic", "intense", "powerful", "striking"],
            "peaceful": ["peaceful", "calm", "serene", "gentle", "soft"],
            "energetic": ["energetic", "dynamic", "vibrant", "lively"],
            "professional": ["professional", "clean", "crisp", "premium"],
            "warm": ["warm", "cozy", "inviting", "friendly"],
            "cool": ["cool", "sleek", "modern", "minimalist"]
        }
        
        for mood, keywords in mood_keywords.items():
            if any(keyword in description.lower() for keyword in keywords):
                elements["mood"] = mood
                break
        
        # Extract environment
        env_keywords = {
            "studio": ["studio", "backdrop", "clean background", "white background"],
            "outdoor": ["outdoor", "nature", "landscape", "sky"],
            "urban": ["urban", "city", "street", "building"],
            "home": ["home", "living room", "kitchen", "interior"],
            "workplace": ["office", "workplace", "desk", "professional environment"]
        }
        
        for env, keywords in env_keywords.items():
            if any(keyword in description.lower() for keyword in keywords):
                elements["environment"] = env
                break
        
        # Extract lighting
        light_keywords = {
            "bright": ["bright", "well-lit", "high-key", "daylight"],
            "dramatic": ["dramatic lighting", "contrast", "shadows", "low-key"],
            "warm": ["warm lighting", "golden hour", "sunset", "amber"],
            "cool": ["cool lighting", "blue tones", "moonlight", "night"],
            "natural": ["natural light", "soft light", "diffused"]
        }
        
        for light, keywords in light_keywords.items():
            if any(keyword in description.lower() for keyword in keywords):
                elements["lighting"] = light
                break
        
        narrative_elements[i] = elements
    
    return narrative_elements

def _validate_and_enhance_transition(transition_prompt, source_elements, target_elements, product_info):
    """Validate and enhance a transition prompt if needed.
    
    Args:
        transition_prompt: Generated transition prompt
        source_elements: Narrative elements of source keyframe
        target_elements: Narrative elements of target keyframe
        product_info: Product information dictionary
        
    Returns:
        str: Validated and potentially enhanced transition prompt
    """
    # Check for error messages or generic content
    error_indicators = [
        "I apologize", "technical issue", "I'm sorry", 
        "cannot generate", "unable to provide"
    ]
    
    if any(indicator in transition_prompt for indicator in error_indicators) or len(transition_prompt) < 50:
        # Create an enhanced transition based on narrative elements
        transition_types = {
            # Dramatic shifts
            ("dramatic", "peaceful"): "gradual dissolve with softening lighting",
            ("peaceful", "dramatic"): "quick zoom with intensifying contrast",
            
            # Environment shifts
            ("studio", "outdoor"): "expanding pull-out shot revealing wider environment",
            ("outdoor", "studio"): "focusing zoom with background blur transition",
            
            # Lighting shifts
            ("bright", "dramatic"): "shadow sweep transition with contrast increase",
            ("warm", "cool"): "color temperature shift dissolve"
        }
        
        # Determine transition type based on narrative shift
        transition_type = "smooth crossfade"  # default
        key = (source_elements["mood"], target_elements["mood"])
        if key in transition_types:
            transition_type = transition_types[key]
        else:
            key = (source_elements["environment"], target_elements["environment"])
            if key in transition_types:
                transition_type = transition_types[key]
            else:
                key = (source_elements["lighting"], target_elements["lighting"])
                if key in transition_types:
                    transition_type = transition_types[key]
        
        # Create enhanced prompt
        enhanced_prompt = (
            f"Create a {transition_type} transition between these two marketing scenes for the {product_info['type']}. "
            f"Start with a {source_elements['mood']} mood in a {source_elements['environment']} setting with "
            f"{source_elements['lighting']} lighting. Transition to a {target_elements['mood']} mood in a "
            f"{target_elements['environment']} setting with {target_elements['lighting']} lighting. "
            f"Keep the {product_info['type']} as the central focus throughout the transition. "
            f"Use a duration of 2-3 seconds with smooth camera movement and professional visual effects "
            f"typical of high-end commercials. Maintain brand consistency and marketing narrative continuity."
        )
        
        return enhanced_prompt
    
    return transition_prompt

def run_full_pipeline(config, image_path, num_keyframes=5, output_dir=None, seed=None):
    """Run the full advertisement generation pipeline with product recognition.
    
    Args:
        config: Configuration object
        image_path: Path to the product image
        num_keyframes: Number of keyframes to generate
        output_dir: Optional directory to save the output
        seed: Random seed for reproducibility
    
    Returns:
        str: Path to the generated advertisement video
    """
    # Create a timestamp-based directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir:
        run_dir = os.path.join(output_dir, f"ad_generation_{timestamp}")
    else:
        run_dir = os.path.join(config.output_dir, f"ad_generation_{timestamp}")
    
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Starting full advertisement generation pipeline. Output dir: {run_dir}")
    
    # Set the random seed if provided
    if seed is not None:
        np.random.seed(seed)
        logger.info(f"Using random seed: {seed}")
        # Save the seed for reproducibility
        with open(os.path.join(run_dir, "seed.txt"), "w") as f:
            f.write(str(seed))
    
    # Save a copy of the input image
    product_image = Image.open(image_path).convert("RGB")
    product_image.save(os.path.join(run_dir, "product_image.jpg"))
    
    # Step 0: Analyze product image with Qwen to identify the product
    logger.info("Step 0: Analyzing product image to identify the product...")
    qwen_model = QwenModel(config)
    product_analysis_prompt = (
        "Analyze this item in the image and provide the following information in a structured format:\n"
        "1. Product Type: What type of product/item is this?\n"
        "2. Key Features: List 3-5 distinctive features of this item\n"
        "3. Brand (if visible): Is there any visible branding?\n"
        "4. Target Audience: Who would be most interested in this item?\n"
        "5. Unique Selling Points: What makes this item special or desirable?\n"
        "6. Suggested Use Context: In what settings or situations would this item be most valuable?\n\n"
        "Be specific and detailed in your analysis, focusing on what makes this item marketable."
    )
    
    product_analysis = qwen_model.generate(
        system_prompt="You are an expert product analyst who can identify and describe products accurately.",
        user_prompt=product_analysis_prompt,
        image=product_image
    )
    
    # Save the product analysis
    with open(os.path.join(run_dir, "product_analysis.txt"), "w", encoding="utf-8") as f:
        f.write(product_analysis)
    
    logger.info(f"Product analysis complete. Extracted product information.")
    
    # Extract product type and key features for use in prompts
    product_info = {
        "type": "product",
        "features": [],
        "brand": "",
        "context": ""
    }
    
    # Parse the product analysis to extract key information
    try:
        if "Product Type:" in product_analysis:
            product_type_section = product_analysis.split("Product Type:")[1].split("\n")[0]
            product_info["type"] = product_type_section.strip()
        
        if "Key Features:" in product_analysis:
            features_section = product_analysis.split("Key Features:")[1].split("Brand")[0]
            features_lines = [line.strip() for line in features_section.split("\n") if line.strip()]
            product_info["features"] = features_lines
        
        if "Brand" in product_analysis:
            brand_section = product_analysis.split("Brand")[1].split("Target Audience:")[0]
            product_info["brand"] = brand_section.strip()
        
        if "Suggested Use Context:" in product_analysis:
            context_section = product_analysis.split("Suggested Use Context:")[1]
            product_info["context"] = context_section.strip()
    except Exception as e:
        logger.warning(f"Error parsing product analysis: {str(e)}. Using default product info.")
    
    # Save the extracted product info
    with open(os.path.join(run_dir, "product_info.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(product_info, f, indent=2)
    
    # Step 1: Generate keyframe prompts with director agent, enriched with product info
    logger.info("Step 1: Generating keyframe prompts with product context...")
    director = DirectorAgent(config)
    
    # Enhance the director agent with product context
    director_context = (
        f"The product is a {product_info['type']}. "
        f"Key features include: {', '.join(product_info['features'][:3])}. "
        f"Brand: {product_info['brand']}. "
        f"Typical use context: {product_info['context']}."
    )
    
    keyframe_prompts = director.create_keyframe_prompts(
        product_image, 
        num_keyframes,
        additional_context=director_context
    )
    
    # Save the keyframe prompts
    keyframes_dir = os.path.join(run_dir, "keyframes")
    os.makedirs(keyframes_dir, exist_ok=True)
    with open(os.path.join(keyframes_dir, "keyframe_prompts.txt"), "w", encoding="utf-8") as f:
        for i, prompt in enumerate(keyframe_prompts):
            f.write(f"KEYFRAME {i+1}:\n{prompt}\n\n")
    
    # Step 2: Generate keyframe images using OmniControl
    logger.info("Step 2: Generating keyframe images...")
    image_agent = ImageAgent(config)
    keyframe_images = []

    # Process keyframe prompts to make them suitable for OmniControl
    processed_prompts = []

    for i, prompt in enumerate(keyframe_prompts):
        # Extract the core creative description
        core_description = ""
        
        # Simple parsing - try to extract the creative part
        if ":" in prompt:
            parts = prompt.split(":", 1)
            # Take what's after the colon
            description = parts[1].strip()
            
            # If there are more detailed sections, just take the first paragraph
            if "\n" in description:
                first_paragraph = description.split("\n")[0].strip()
                core_description = first_paragraph
            else:
                core_description = description
        else:
            core_description = prompt
        
        # Simplify and clean up the description - keep only first 1-2 sentences
        if "." in core_description:
            sentences = core_description.split(".")
            if len(sentences) > 2:
                core_description = ".".join(sentences[:2]) + "."
        
        # Make sure it's not too long
        # if len(core_description) > 100:
        #     core_description = core_description[:100].strip() + "..."
        
        # Modified here: Use regular expression to replace the entire word, avoiding partial replacements
        # Ensure replacement is for the complete product name, not part of a word
        if product_info['type']:
            # Use word boundaries (\b) to ensure only complete words are replaced
            core_description = re.sub(r'\b' + re.escape(product_info['type']) + r'\b', 
                                    "this item", 
                                    core_description, 
                                    flags=re.IGNORECASE)
        
        # Ensure no duplicate "this item" is inserted
        core_description = core_description.replace("this item this item", "this item")
        
        # Create a simple creative prompt format
        processed_prompt = f"{core_description} This item is a {product_info['type']}."
        processed_prompt = core_description
        # Log the simplified prompt
        logger.info(f"Keyframe {i+1} simplified prompt: {processed_prompt}")
        
        processed_prompts.append(processed_prompt)
        
        # Save the processed prompt
        with open(os.path.join(keyframes_dir, f"keyframe_{i+1}_processed_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(processed_prompt)

    # Generate each keyframe image
    keyframe_images = []
    for i, processed_prompt in enumerate(processed_prompts):
        logger.info(f"  Generating keyframe {i+1}/{len(processed_prompts)}...")
        
        try:
            keyframe_seed = np.random.randint(0, 1000000) if seed is None else seed + i
            
            # Set reference weight - from high to low
            reference_weight = max(0.9 - (i * 0.1), 0.3)  # Decrease weight, but maintain a minimum value
            
            logger.info(f"  Using product image as reference with weight {reference_weight}")
            
            # Attempt to generate with reference image
            try:
                keyframe = image_agent.generate_with_reference(
                    processed_prompt, 
                    product_image,  # Always use the product image as reference 
                    keyframe_seed,
                    reference_weight=reference_weight
                )
                logger.info(f"  Keyframe {i+1} generated successfully with reference")
            except Exception as e:
                logger.error(f"  Error generating keyframe {i+1} with reference: {str(e)}")
                logger.info(f"  Attempting to generate keyframe {i+1} without reference")
                
                # Fall back to generation without reference
                keyframe = image_agent.generate(processed_prompt, keyframe_seed)
                logger.info(f"  Keyframe {i+1} generated without reference")
            
            keyframe_images.append(keyframe)
            keyframe.save(os.path.join(keyframes_dir, f"keyframe_{i+1}.png"))
            
        except Exception as e:
            logger.error(f"  Failed to generate keyframe {i+1}: {str(e)}")
            
            # Create a blank or placeholder image
            placeholder_size = (512, 512)
            placeholder = Image.new('RGB', placeholder_size, (200, 200, 200))
            
            # Use text to indicate this is a placeholder image
            if PIL_DRAW_AVAILABLE:
                draw = ImageDraw.Draw(placeholder)
                try:
                    # Attempt to use system font
                    font = ImageFont.truetype("Arial", 20)
                except:
                    # Fall back to default font
                    font = ImageFont.load_default()
                
                draw.text((50, 200), f"Keyframe {i+1} generation failed", fill=(0, 0, 0), font=font)
                draw.text((50, 230), "Using placeholder image", fill=(0, 0, 0), font=font)
            
            keyframe_images.append(placeholder)
            placeholder.save(os.path.join(keyframes_dir, f"keyframe_{i+1}_placeholder.png"))
            
            # Log to file
            with open(os.path.join(keyframes_dir, f"keyframe_{i+1}_error.txt"), "w", encoding="utf-8") as f:
                f.write(f"Error generating keyframe {i+1}: {str(e)}\n")
                f.write(f"Prompt: {processed_prompt}\n")
    # Step 3: Generate transition prompts with product context
    logger.info("Step 3: Generating transition prompts...")
    transition_agent = TransitionAgent(config)
    transitions_dir = os.path.join(run_dir, "transitions")
    os.makedirs(transitions_dir, exist_ok=True)
    
    # Extract narrative elements from keyframe prompts
    narrative_elements = _extract_narrative_elements(keyframe_prompts)

    # Generate transition prompts with narrative context
    transition_prompts = []
    for i in range(len(keyframe_images) - 1):
        logger.info(f"  Generating transition prompt {i+1} -> {i+2}...")
        
        # Ensure we have valid PIL images
        source_image = keyframe_images[i]
        target_image = keyframe_images[i+1]
        
        # Check image objects
        if not isinstance(source_image, Image.Image) or not isinstance(target_image, Image.Image):
            logger.error(f"Invalid keyframe image types: source={type(source_image)}, target={type(target_image)}")
            # Create a default transition
            transition_prompt = f"Smooth crossfade transition from keyframe {i+1} to keyframe {i+2}, gradually blending elements over 2 seconds."
        else:
            # Get descriptions
            source_prompt = keyframe_prompts[i]
            target_prompt = keyframe_prompts[i+1]
            
            # Get narrative elements
            source_elements = narrative_elements.get(i, {"mood": "neutral", "environment": "studio", "lighting": "standard"})
            target_elements = narrative_elements.get(i+1, {"mood": "neutral", "environment": "studio", "lighting": "standard"})
            
            # Create specialized transition context
            transition_context = (
                f"This transition moves from a {source_elements['mood']} scene in a "
                f"{source_elements['environment']} with {source_elements['lighting']} lighting "
                f"to a {target_elements['mood']} scene in a {target_elements['environment']} "
                f"with {target_elements['lighting']} lighting. "
                f"This is a {product_info['type']} marketing video that should maintain professional quality."
            )
            
            # Call the transition agent
            try:
                transition_prompt = transition_agent.create_transition_prompt(
                    source_image,
                    target_image,
                    source_prompt,
                    target_prompt,
                    product_context=transition_context
                )
            except Exception as e:
                logger.error(f"Error generating transition {i+1}->{i+2}: {str(e)}")
                # Create a fallback transition
                transition_prompt = f"Smooth crossfade transition from keyframe {i+1} to keyframe {i+2}, gradually blending elements over 2 seconds."
        
        transition_prompts.append(transition_prompt)
        
        # Save transition prompt
        with open(os.path.join(transitions_dir, f"transition_{i+1}_to_{i+2}.txt"), "w", encoding="utf-8") as f:
            f.write(transition_prompt)
    
    # Step 4: Generate transition videos and assemble final video
    logger.info("Step 4: Generating transitions and assembling final video...")
    video_agent = VideoAgent(config)
    
    # Generate transition videos
    transition_clips = []
    for i in range(len(keyframe_images) - 1):
        logger.info(f"  Generating transition video {i+1} -> {i+2}...")
        source_image = keyframe_images[i]
        target_image = keyframe_images[i+1]
        transition_prompt = transition_prompts[i]
        
        transition_clip = video_agent.generate_transition(
            source_image, target_image, transition_prompt
        )
        
        transition_clips.append(transition_clip)
        transition_clip.save(os.path.join(transitions_dir, f"transition_{i+1}_to_{i+2}.mp4"))
    
    # Assemble final video
    logger.info("Assembling final advertisement video...")
    final_video = video_agent.assemble_final_video(keyframe_images, transition_clips)
    
    # Save the final video
    final_video_path = os.path.join(run_dir, "final_advertisement.mp4")
    final_video.save(final_video_path)
    
    # Create a symlink or copy to the latest output for easy access
    latest_link = os.path.join(config.output_dir, "latest_advertisement.mp4")
    if os.path.exists(latest_link):
        os.remove(latest_link)
    
    try:
        os.symlink(final_video_path, latest_link)
    except (OSError, AttributeError):
        # If symlinks not supported (e.g., on Windows), make a copy
        shutil.copy(final_video_path, latest_link)
    
    logger.info(f"Advertisement generation complete! Final video saved to: {final_video_path}")
    return final_video_path

def main():
    """Main function that handles all operation modes."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Load configuration
        config = Config()
        logger.info(f"Loaded configuration")
        
        # Check which mode to run
        if args.mode == 'pipeline':
            logger.info("Running full advertisement generation pipeline...")
            run_full_pipeline(
                config, 
                args.image_path, 
                args.keyframes, 
                args.output_dir, 
                args.seed
            )
            
        elif args.mode == 'image':
            logger.info("Running image generation test...")
            test_image_generation(
                config, 
                args.image_path, 
                args.prompt, 
                args.seed
            )
            
        elif args.mode == 'qwen':
            logger.info("Running Qwen model test...")
            test_qwen_model(
                config, 
                args.image_path,
                args.prompt
            )
            
        elif args.mode == 'director':
            logger.info("Running director agent test...")
            test_director_agent(
                config, 
                args.image_path, 
                args.keyframes
            )
            
        elif args.mode == 'transition':
            logger.info("Running transition agent test...")
            test_transition_agent(
                config, 
                args.source_image, 
                args.target_image, 
                args.source_prompt, 
                args.target_prompt
            )
            
        elif args.mode == 'video':
            logger.info("Running video agent test...")
            test_video_agent(
                config, 
                args.keyframes_dir, 
                args.transitions_dir, 
                args.output_path
            )
            
        else:
            logger.error(f"Unknown mode: {args.mode}")
            print(f"Please specify a valid mode. Use --help for more information.")
            return 1
        
        logger.info("Operation completed successfully")
        return 0
    
    except Exception as e:
        logger.exception(f"Error in main: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)