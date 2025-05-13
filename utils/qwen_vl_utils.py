#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for Qwen2.5-VL model.
"""

from PIL import Image

def process_vision_info(messages):
    """Process vision information (images and videos) from messages.
    
    Args:
        messages: List of message dictionaries containing roles and content
        
    Returns:
        tuple: (image_inputs, video_inputs)
    """
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        if "content" in message:
            content = message["content"]
            
            # Handle single content item
            if isinstance(content, dict):
                content = [content]
                
            # Process list of content items
            for item in content:
                if isinstance(item, dict):
                    # Process image
                    if item.get("type") == "image" and "image" in item:
                        image_data = item["image"]
                        # If image_data is already a PIL.Image, use it directly
                        if isinstance(image_data, Image.Image):
                            image_inputs.append(image_data)
                    
                    # Process video (if needed in the future)
                    elif item.get("type") == "video" and "video" in item:
                        # Not implemented in this basic version
                        pass
                        
    return image_inputs, video_inputs