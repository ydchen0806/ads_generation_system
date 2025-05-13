import torch
from typing import Optional, Union, List, Tuple
from diffusers.pipelines import FluxPipeline
from PIL import Image, ImageFilter
import numpy as np
import cv2

from .pipeline_tools import encode_images

condition_dict = {
    "subject": 0,
    "sam": 1,
    "dinov2": 2,
}


class Condition(object):
    def __init__(
        self,
        condition_type: str,
        raw_img: Union[Image.Image, torch.Tensor] = None,
        condition: Union[Image.Image, torch.Tensor] = None,
        mask=None,
        position_delta=None,
        position_scale=1.0,
        crop_region: List[int] = None,  # [x1, y1, x2, y2]
    ) -> None:
        self.condition_type = condition_type
        assert raw_img is not None or condition is not None
        if raw_img is not None:
            self.condition = self.get_condition(condition_type, raw_img)
        else:
            self.condition = condition
        self.position_delta = position_delta
        self.position_scale = position_scale
        self.crop_region = crop_region
        # TODO: Add mask support
        assert mask is None, "Mask not supported yet"
        
        if self.crop_region is not None and raw_img is not None:
            if isinstance(raw_img, Image.Image):
                raw_img = raw_img.crop(self.crop_region)
            elif isinstance(raw_img, torch.Tensor):
                x1, y1, x2, y2 = self.crop_region
                raw_img = raw_img[..., y1:y2, x1:x2]

    def get_condition(
        self, condition_type: str, raw_img: Union[Image.Image, torch.Tensor]
    ) -> Union[Image.Image, torch.Tensor]:
        """
        Returns the condition image.
        """
        if condition_type == "subject":
            return raw_img
        elif condition_type == "sam":
            from segment_anything import SamPredictor, sam_model_registry
            sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to("cuda")
            predictor = SamPredictor(sam)
            img = np.array(raw_img.convert("RGB"))
            predictor.set_image(img)
            masks, _, _ = predictor.predict()
            # Enhance subject features with SAM masks
            if isinstance(raw_img, Image.Image):
                raw_img = np.array(raw_img)
            enhanced_img = raw_img * masks[0][..., None]
            return torch.from_numpy(enhanced_img)
        elif condition_type == "dinov2":
            from transformers import AutoImageProcessor, AutoModel
            processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
            model = AutoModel.from_pretrained("facebook/dinov2-base").to("cuda")
            inputs = processor(images=raw_img, return_tensors="pt").to("cuda")
            outputs = model(**inputs)
            # Combine DINOv2 features with original image
            if isinstance(raw_img, Image.Image):
                raw_img = np.array(raw_img)
            features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            enhanced_img = raw_img * features[..., None]
            return torch.from_numpy(enhanced_img)
        return self.condition

    @property
    def type_id(self) -> int:
        """
        Returns the type id of the condition.
        """
        return condition_dict[self.condition_type]

    @classmethod
    def get_type_id(cls, condition_type: str) -> int:
        """
        Returns the type id of the condition.
        """
        return condition_dict[condition_type]

    def encode(self, pipe: FluxPipeline) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Encodes the condition into tokens, ids and type_id.
        """
        if self.condition_type in ["subject", "sam", "dinov2"]:
            tokens, ids = encode_images(pipe, self.condition)
        else:
            raise NotImplementedError(
                f"Condition type {self.condition_type} not implemented"
            )
        if self.position_delta is None and self.condition_type == "subject":
            self.position_delta = [0, -self.condition.size[0] // 16]
        if self.position_delta is not None:
            ids[:, 1] += self.position_delta[0]
            ids[:, 2] += self.position_delta[1]
        if self.position_scale != 1.0:
            scale_bias = (self.position_scale - 1.0) / 2
            ids[:, 1] *= self.position_scale
            ids[:, 2] *= self.position_scale
            ids[:, 1] += scale_bias
            ids[:, 2] += scale_bias
        type_id = torch.ones_like(ids[:, :1]) * self.type_id
        return tokens, ids, type_id
