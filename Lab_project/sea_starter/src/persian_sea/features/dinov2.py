from __future__ import annotations

from contextlib import nullcontext
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DinoV2FeatureExtractor:
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: str = "auto",
        amp: bool = True,
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.amp = bool(amp and self.device.type == "cuda")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False

        self.hidden_size = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode(self, images: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        if not images:
            return np.empty((0, self.hidden_size), dtype=np.float32)

        outputs = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            pil = [Image.fromarray(img.astype(np.uint8), mode="RGB") for img in batch]
            inputs = self.processor(images=pil, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if self.amp
                else nullcontext()
            )
            with autocast:
                out = self.model(**inputs)
                # DINOv2 [CLS] representation. Shape [B, hidden].
                feat = out.last_hidden_state[:, 0]
                feat = torch.nn.functional.normalize(feat.float(), dim=-1)

            outputs.append(feat.cpu().numpy())

        return np.concatenate(outputs, axis=0).astype(np.float32)
