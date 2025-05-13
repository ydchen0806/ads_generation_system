#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for image processing.
"""

from PIL import Image
import os

def load_image(path):
    """Load an image from file.
    
    Args:
        path: Path to the image file
        
    Returns:
        PIL.Image: Loaded image
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    try:
        image = Image.open(path)
        return image
    except Exception as e:
        raise ValueError(f"Error loading image: {str(e)}")

def resize_image(image, target_size):
    """Resize an image to target dimensions.
    
    Args:
        image: PIL Image to resize
        target_size: (width, height) tuple
        
    Returns:
        PIL.Image: Resized image
    """
    return image.resize(target_size, Image.LANCZOS)

def crop_to_aspect_ratio(image, aspect_ratio):
    """Crop an image to match the specified aspect ratio.
    
    Args:
        image: PIL Image to crop
        aspect_ratio: Target aspect ratio (width/height)
        
    Returns:
        PIL.Image: Cropped image
    """
    width, height = image.size
    current_ratio = width / height
    
    if current_ratio > aspect_ratio:
        # Image is wider than target ratio, crop width
        new_width = int(height * aspect_ratio)
        left = (width - new_width) // 2
        return image.crop((left, 0, left + new_width, height))
    elif current_ratio < aspect_ratio:
        # Image is taller than target ratio, crop height
        new_height = int(width / aspect_ratio)
        top = (height - new_height) // 2
        return image.crop((0, top, width, top + new_height))
    else:
        # Aspect ratio already matches
        return image