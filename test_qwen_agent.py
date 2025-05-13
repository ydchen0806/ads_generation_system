# test_qwen.py

import os
import logging
import sys
from PIL import Image
from models.qwen_connector import QwenModel

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)

# 创建一个简单的配置对象
class SimpleConfig:
    def __init__(self):
        self.qwen_model_path = "/h3cstore_ns/ydchen/code/SubjectGenius/model/Qwen2.5-VL-7B-Instruct"
        self.temp_dir = "/tmp"
        self.max_tokens = 512

# 创建 QwenModel 实例
config = SimpleConfig()
qwen_model = QwenModel(config)

# 测试单个图像
test_image_path = "/h3cstore_ns/ydchen/code/wan_2_1/ads_generation_system/test_objects0507/cup1.jpg"  # 更改为实际图像路径
if os.path.exists(test_image_path):
    image = Image.open(test_image_path)
    
    # 测试产品分析
    system_prompt = "You are an expert product analyst."
    user_prompt = "Analyze this product image and describe its key features."
    
    response = qwen_model.generate(system_prompt, user_prompt, image=image)
    print("\n=== Product Analysis ===")
    print(response)
    
    # 测试关键帧生成
    system_prompt = "You are an advertising creative director."
    user_prompt = "Create 3 keyframe descriptions for a marketing video about this item."
    
    response = qwen_model.generate(system_prompt, user_prompt, image=image)
    print("\n=== Keyframe Descriptions ===")
    print(response)
else:
    print(f"Test image not found at {test_image_path}")