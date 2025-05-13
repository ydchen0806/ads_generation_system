#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Templates for prompts used by the agents.
"""

def get_director_prompt(num_keyframes=5):
    """Get the prompt for the director agent.
    
    Args:
        num_keyframes: Number of keyframes to generate
        
    Returns:
        str: Director prompt
    """
    return f"""
You are tasked with creating {num_keyframes} distinct creative scene descriptions for marketing images featuring this item shown in the image.

For each scene:
1. Create a SHORT, CREATIVE description (1-2 sentences only)
2. Each description should place this item in a unique, interesting scenario or setting
3. Use descriptive but concise language
4. Include interesting lighting, weather, or atmospheric elements
5. Describe the item's position, surroundings, or interaction with the environment
6. In at least 2 scenes, this item must visibly interact with a character (e.g., being held, used, or admired)
7. Refer to the product as "this item" rather than repeating its name

Examples of good descriptions:
- "On a foggy autumn morning, this item sits on a moss-covered rock by a mountain stream."
- "This item glows under dramatic stage lighting, held aloft by a dancer amid falling rose petals."
- "Suspended in mid-air against a sunset backdrop, this item casts a long shadow on desert sand."
- "In a bustling market, a merchant showcases this item under the golden glow of lanterns."

BAD example (too generic): 
- "Product shot with professional lighting and attractive background."

Format your response as {num_keyframes} numbered scenes with just the creative description.
Do not include additional explanations, purposes, or technical details.
Keep each description under 100 characters if possible.
"""

def get_transition_prompt(source_prompt, target_prompt):
    """Get the prompt for the transition agent.
    
    Args:
        source_prompt: Source keyframe prompt
        target_prompt: Target keyframe prompt
        
    Returns:
        str: Transition prompt
    """
    # Extract titles or first parts for brief descriptions
    source_brief = source_prompt.split(":")[0] if ":" in source_prompt else "first scene"
    target_brief = target_prompt.split(":")[0] if ":" in target_prompt else "second scene"
    
    return f"""
I need to create a smooth and professional transition between two keyframes in a marketing video.

KEYFRAME 1: {source_brief}
Full description: {source_prompt}

KEYFRAME 2: {target_brief}
Full description: {target_prompt}

Create a detailed description of how to transition from Keyframe 1 to Keyframe 2. 
Your description should:

1. Specify the exact camera movement (pan, zoom, rotate, dolly, etc.)
2. Describe any visual effects to use (dissolves, fades, morphs, wipes, etc.)
3. Detail how elements from both frames interact during the transition
4. Specify timing and pacing (how many seconds, acceleration/deceleration)
5. Include any lighting changes, color shifts, or atmosphere transitions
6. Maintain focus on the product throughout the transition

Make your transition emotionally engaging and dramatically appropriate for the shift in narrative between these scenes.
The transition should feel like a premium commercial for a high-end product.

Be specific and technical, as if giving instructions to a professional video editor.
"""