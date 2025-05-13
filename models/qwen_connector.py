# models/qwen_connector.py

import logging
import os
from io import BytesIO
import base64
from PIL import Image
import torch
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

logger = logging.getLogger(__name__)

class QwenModel:
    """Connector for the locally deployed Qwen2.5-VL model."""
    
    def __init__(self, config):
        """Initialize the model with configuration settings.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model_path = getattr(config, 'qwen_model_path', "/h3cstore_ns/ydchen/code/SubjectGenius/model/Qwen2.5-VL-7B-Instruct")
        self.device = getattr(config, 'device', "cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = getattr(config, 'max_tokens', 512)
        
        logger.info(f"Initializing Qwen2.5-VL model from {self.model_path}")
        try:
            # Load the model
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
                device_map="auto"
            )
            
            # Load the processor
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
            logger.info("Qwen2.5-VL model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Qwen2.5-VL model: {str(e)}")
            logger.warning("Initializing in mock mode due to model loading error")
            self.model = None
            self.processor = None
    
    def generate(self, system_prompt, user_prompt, image=None, images=None):
        """Generate text using the Qwen2.5-VL model.
        
        Args:
            system_prompt: System prompt text
            user_prompt: User prompt text
            image: Optional single image (PIL Image)
            images: Optional list of image objects with descriptions
            
        Returns:
            str: Generated text
        """
        if self.model is None or self.processor is None:
            logger.warning("Model not loaded. Using mock response.")
            return self._generate_mock_response(system_prompt, user_prompt, image, images)
        
        try:
            # Prepare message list
            messages = [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}]
            
            # Add user message content
            user_content = []
            
            # Add text prompt
            user_content.append({"type": "text", "text": user_prompt})
            
            # Process single image
            if image is not None:
                logger.info("Processing single image")
                # Verify image is a PIL Image
                if not isinstance(image, Image.Image):
                    logger.error(f"Invalid image type: {type(image)}")
                    raise ValueError(f"Expected PIL Image, got {type(image)}")
                
                # Save image to temporary file (qwen_vl_utils may require file path)
                temp_img_path = os.path.join(
                    getattr(self.config, 'temp_dir', '/tmp'), 
                    f"qwen_img_{np.random.randint(10000)}.jpg"
                )
                os.makedirs(os.path.dirname(temp_img_path), exist_ok=True)
                
                # Save as JPEG
                image.save(temp_img_path, format="JPEG")
                
                # Add image to content
                user_content.append({
                    "type": "image",
                    "image": temp_img_path,
                })
            
            # Process multiple images
            if images is not None:
                logger.info(f"Processing {len(images)} images")
                for i, img_obj in enumerate(images):
                    try:
                        img_data = img_obj["data"]
                        if not isinstance(img_data, Image.Image):
                            logger.error(f"Invalid image type in batch: {type(img_data)}")
                            continue
                        
                        # Save image to temporary file
                        temp_img_path = os.path.join(
                            getattr(self.config, 'temp_dir', '/tmp'), 
                            f"qwen_batch_img_{i}_{np.random.randint(10000)}.jpg"
                        )
                        img_data.save(temp_img_path, format="JPEG")
                        
                        # Add image to content
                        user_content.append({
                            "type": "image",
                            "image": temp_img_path,
                        })
                        
                        # Add description if provided
                        if "description" in img_obj and img_obj["description"]:
                            user_content.append({
                                "type": "text", 
                                "text": f"[Image {i+1}: {img_obj['description']}]"
                            })
                    except Exception as e:
                        logger.error(f"Error processing image {i} in batch: {str(e)}")
            
            # Add user message
            messages.append({"role": "user", "content": user_content})
            
            # Prepare for inference
            logger.info("Preparing for inference")
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # Move inputs to the appropriate device
            inputs = inputs.to(self.device)
            
            # Generate output
            logger.info("Generating text")
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            # Clean up temporary files
            self._cleanup_temp_files()
            
            return output_text[0] if output_text else ""
        
        except Exception as e:
            logger.error(f"Error generating text with Qwen2.5-VL: {str(e)}")
            logger.exception("Full traceback:")
            return f"I apologize, but I'm having technical difficulties processing the images or generating appropriate text."
        
    def _cleanup_temp_files(self):
        """Remove temporary files created during processing."""
        temp_dir = getattr(self.config, 'temp_dir', '/tmp')
        try:
            # Delete temporary files starting with qwen_img_ or qwen_batch_img_
            for filename in os.listdir(temp_dir):
                if filename.startswith("qwen_img_") or filename.startswith("qwen_batch_img_"):
                    try:
                        os.remove(os.path.join(temp_dir, filename))
                    except:
                        pass
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {str(e)}")

    def _generate_mock_response(self, system_prompt, user_prompt, image=None, images=None):
        """Generate a mock response when the model is unavailable.
        
        Args:
            system_prompt: System prompt text
            user_prompt: User prompt text
            image: Optional single image
            images: Optional list of image objects with descriptions
            
        Returns:
            str: Mock response
        """
        logger.warning("Using mock response generator")
        # Check for keywords in the prompt to generate relevant response
        prompt_lower = user_prompt.lower()
        
        # Detect keyframe generation request
        if "keyframe" in prompt_lower or "scene" in prompt_lower:
            return self._generate_mock_keyframes()
        
        # Detect transition request
        if "transition" in prompt_lower:
            return self._generate_mock_transition()
        
        # Detect product analysis request
        if "analyze" in prompt_lower and "product" in prompt_lower:
            return self._generate_mock_product_analysis()
        
        # Default response
        return (
            "Based on the information provided, I've created a detailed response "
            "that addresses your request. The content focuses on the key elements "
            "you mentioned while maintaining a professional and engaging tone."
        )

    def _generate_mock_keyframes(self):
        """Generate mock keyframe descriptions."""
        keyframes = [
            "Scene 1: Product displayed on a pedestal with dramatic spotlight, casting bold shadows against a dark background.",
            "Scene 2: Product held in a person's hand, demonstrating its scale and utility in a bright, modern kitchen.",
            "Scene 3: Product placed on a wooden table by a window, with soft morning light accentuating its texture and design.",
            "Scene 4: Close-up of product with water droplets on its surface, highlighting material quality and durability.",
            "Scene 5: Product integrated into a lifestyle setting with happy users, showcasing its benefits in a social context."
        ]
        
        return "\n\n".join(keyframes)

    def _generate_mock_transition(self):
        """Generate mock transition descriptions."""
        transitions = [
            "For this transition, use a smooth crossfade blending from the first scene to the second over 2 seconds. This subtle transition keeps the product in focus while shifting from a dramatic spotlight to a practical usage scenario.",
            "Create a dynamic transition with the camera orbiting the product as the background changes from a kitchen to a natural morning light scene over 3 seconds. This keeps the product central while the environment transforms.",
            "Use a splash transition where water droplets ripple across the screen, washing away the current scene to reveal the next. This ties thematically to the water droplets in the destination scene, creating a memorable visual link."
        ]
        
        import random
        return random.choice(transitions)

    def _generate_mock_product_analysis(self):
        """Generate mock product analysis."""
        return """
    Product Type: Modern portable water bottle

    Key Features:
    1. Double-walled insulation for temperature retention
    2. Sleek minimalist design with matte finish
    3. Leak-proof lid with convenient carrying handle
    4. Made from eco-friendly, sustainable materials
    5. Available in multiple color options

    Brand (if visible): EcoHydrate

    Target Audience: Active professionals, fitness enthusiasts, environmentally conscious consumers, and style-oriented individuals valuing form and function.

    Suggested Use Context: Daily use for work or gym, outdoor activities, travel, and as a stylish hydration accessory.
    """