import lightning as L
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from transformers import pipeline
import cv2
import torch
import os

try:
    import wandb
except ImportError:
    wandb = None

from ..flux.condition import Condition
from ..flux.generate import generate


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            self.generate_a_sample(
                trainer,
                pl_module,
                f"{self.save_path}/{self.run_name}/output",
                f"lora_{self.total_steps}",
                batch["condition_type"][
                    0
                ],  # Use the condition type from the current batch
            )

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        save_path,
        file_name,
        condition_type="super_resolution",
    ):
        # TODO: change this two variables to parameters
        condition_size = trainer.training_config["dataset"]["condition_size"]
        target_size = trainer.training_config["dataset"]["target_size"]
        position_scale = trainer.training_config["dataset"].get("position_scale", 1.0)

        generator = torch.Generator(device=pl_module.device)
        generator.manual_seed(42)

        test_list = []

        if condition_type == "subject":
            test_list.extend(
                [
                    (
                        Image.open("assets/test_in.jpg"),
                        [0, -32],
                        "Resting on the picnic table at a lakeside campsite, it's caught in the golden glow of early morning, with mist rising from the water and tall pines casting long shadows behind the scene.",
                    ),
                    (
                        Image.open("assets/test_out.jpg"),
                        [0, -32],
                        "In a bright room. It is placed on a table.",
                    ),
                ]
            )
            
            # Add SAM and DINOv2 as enhancement options for subject
            if self.training_config.get("use_sam", False):
                from segment_anything import SamPredictor, sam_model_registry
                if not hasattr(self, 'sam_predictor'):
                    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
                    self.sam_predictor = SamPredictor(sam)
                condition_img = Image.open("assets/vase_hq.jpg").resize((condition_size, condition_size)).convert("RGB")
                self.sam_predictor.set_image(np.array(condition_img))
                masks, _, _ = self.sam_predictor.predict()
                condition_img = Image.fromarray(masks[0].astype(np.uint8) * 255).convert("RGB")
                test_list.append((condition_img, [0, 0], "A vase with SAM segmentation."))
                
            if self.training_config.get("use_dinov2", False):
                from transformers import AutoImageProcessor, AutoModel
                if not hasattr(self, 'dinov2_processor'):
                    self.dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
                    self.dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base")
                condition_img = Image.open("assets/vase_hq.jpg").resize((condition_size, condition_size)).convert("RGB")
                inputs = self.dinov2_processor(images=condition_img, return_tensors="pt")
                outputs = self.dinov2_model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
                condition_img = Image.fromarray((features * 255).astype(np.uint8)).convert("RGB")
                test_list.append((condition_img, [0, 0], "A vase with DINOv2 features."))
        else:
            raise NotImplementedError

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i, (condition_img, position_delta, prompt, *others) in enumerate(test_list):
            condition = Condition(
                condition_type=condition_type,
                condition=condition_img.resize(
                    (condition_size, condition_size)
                ).convert("RGB"),
                position_delta=position_delta,
                crop_region=batch.get("crop_region", None),
                **(others[0] if others else {}),
            )
            res = generate(
                pl_module.flux_pipe,
                prompt=prompt,
                conditions=[condition],
                height=target_size,
                width=target_size,
                generator=generator,
                model_config=pl_module.model_config,
                default_lora=True,
            )
            res.images[0].save(
                os.path.join(save_path, f"{file_name}_{condition_type}_{i}.jpg")
            )
