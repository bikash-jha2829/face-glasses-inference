import logging

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from config.settings import ClipDetectionConfig

logger = logging.getLogger(__name__)

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

# Global cache for the CLIP classifier instance per worker.
_cached_clip_classifier = None


class ClipGlassesClassifier:
    def __init__(self) -> None:
        self.model_name = ClipDetectionConfig["model_name"]
        self.device = DEVICE
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.labels = ClipDetectionConfig["text_descriptions"]

        # Precompute text embeddings once
        text_inputs = self.processor(
            text=self.labels, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            # Normalize text embeddings
            text_embeds = self.model.get_text_features(**text_inputs)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

            # Also cache logit_scale once (it does not change in eval mode)
            logit_scale = self.model.logit_scale.exp()

        self.text_embeds = text_embeds
        self.logit_scale = logit_scale
        self.model.eval()

    def classify_faces_batch(
        self, face_images, return_embeddings: bool = False
    ) -> list:
        """
        Classify a batch of face images, optionally returning their CLIP embeddings.
        For each face crop (224×224 RGB recommended), we:
          1) Compute image embeddings (img_embeds)
          2) Compute similarity with precomputed text_embeds
          3) Apply softmax to get probability distribution

        If return_embeddings=True, we include 'clip_embedding' in the result (float32 list).
        """
        if not face_images:
            return []

        # Preprocess images for CLIP
        inputs = self.processor(
            images=face_images, return_tensors="pt", padding=True
        ).to(self.device)

        with torch.no_grad():
            # (1) Extract and normalize image embeddings
            img_embeds = self.model.get_image_features(**inputs)
            img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)

            # (2) Compute similarity to text embeddings
            similarity = self.logit_scale * (img_embeds @ self.text_embeds.T)
            probs = (
                similarity.softmax(dim=1).cpu().numpy()
            )  # shape: (batch_size, num_labels)

        # Vectorized argmax & max
        pred_idxs = probs.argmax(axis=1)
        confs = probs.max(axis=1)

        # If we need embeddings, move them to CPU and convert to float32
        if return_embeddings:
            img_embeds = img_embeds.cpu().numpy().astype(np.float32)

        results = []
        threshold = ClipDetectionConfig["threshold"]

        for i, row_probs in enumerate(probs):
            pred_idx = pred_idxs[i]
            conf = confs[i]

            # Build dictionary of label -> probability
            metadata = dict(zip(self.labels, map(float, row_probs)))

            if conf < threshold:
                result = {
                    "label": "no confident label",
                    "confidence": float(conf),
                    "clip_metadata": metadata,
                }
            else:
                result = {
                    "label": self.labels[pred_idx],
                    "confidence": float(conf),
                    "clip_metadata": metadata,
                }

            # Optionally attach embeddings
            if return_embeddings:
                result["clip_embedding"] = img_embeds[i].tolist()

            results.append(result)

        return results


def get_clip_classifier() -> ClipGlassesClassifier:
    """
    Return a cached instance of ClipGlassesClassifier per worker.
    """
    global _cached_clip_classifier
    if _cached_clip_classifier is None:
        _cached_clip_classifier = ClipGlassesClassifier()
    return _cached_clip_classifier
