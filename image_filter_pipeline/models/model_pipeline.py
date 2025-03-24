import gc
import io
import json
import logging
import math

import numpy as np
import pandas as pd
from PIL import Image

from models.glasses_classifier import get_clip_classifier
from models.yolo_detector import get_yolo_detector

logger = logging.getLogger(__name__)


def run_inference_partition(
    df_partition: pd.DataFrame, batch_size: int = 4, return_embeddings: bool = False
) -> pd.DataFrame:
    """
    Run YOLO face detection and CLIP classification on a partition of the DataFrame.
    Optionally returns face embeddings if `return_embeddings=True`.
    """

    # We include clip_embedding in our columns so the DataFrame schema is correct
    columns = [
        "image_url",
        "face_confidence",
        "bbox",
        "glasses_label",
        "glasses_confidence",
        "clip_metadata",
        "clip_embedding",
    ]

    if df_partition.empty:
        logger.warning("Partition is empty.")
        return pd.DataFrame(columns=columns)

    yolo = get_yolo_detector()
    clip = get_clip_classifier()

    # 1) Decode image}
    pil_images = []
    image_urls = []
    for idx, row in df_partition.iterrows():
        image_bytes = row.get("image_bytes")
        image_url = row.get("image_url")
        if image_bytes is None:
            pil_images.append(None)
            image_urls.append(image_url)
            continue
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            if pil_img.mode == "P":
                pil_img = pil_img.convert("RGBA")
            if pil_img.mode == "RGBA":
                pil_img = Image.alpha_composite(
                    Image.new("RGB", pil_img.size, (255, 255, 255)), pil_img
                )
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            # Resize to 320x320
            pil_img = pil_img.resize((320, 320), Image.Resampling.BICUBIC)
            pil_images.append(pil_img)
            image_urls.append(image_url)
        except Exception as e:
            logger.warning(f"Could not decode {image_url}. Error: {e}")
            pil_images.append(None)
            image_urls.append(image_url)

    # 2) YOLO face detection
    faces_batch = []
    num_batches = math.ceil(len(pil_images) / batch_size)
    for b in range(num_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, len(pil_images))
        batch_imgs = pil_images[start:end]
        faces_batch.extend(yolo.detect_faces_batch(batch_imgs))

    # 3) Crop faces
    all_face_crops = []
    metadata_list = []
    for idx, faces in enumerate(faces_batch):
        pil_img = pil_images[idx]
        if pil_img is None or not faces:
            continue
        image_url = image_urls[idx]
        for det in faces:
            x1, y1, x2, y2 = det["bbox"]
            cropped = pil_img.crop((x1, y1, x2, y2)).convert("RGB")
            all_face_crops.append(cropped)
            metadata_list.append(
                {
                    "image": pil_img,
                    "image_url": image_url,
                    "face_confidence": det["face_confidence"],
                    "bbox": det["bbox"],
                }
            )

    del pil_images

    # 4) CLIP classification: pass return_embeddings here!
    clip_results = []
    if all_face_crops:
        clip_results = clip.classify_faces_batch(
            all_face_crops, return_embeddings=return_embeddings
        )

    del all_face_crops
    gc.collect()

    # 5) Combine into final DataFrame
    final_rows = []
    for meta, clip_res in zip(metadata_list, clip_results):
        row = {
            "image": meta["image"],
            "image_url": meta["image_url"],
            "face_confidence": meta["face_confidence"],
            "bbox": json.dumps(meta["bbox"]),
            "glasses_label": clip_res["label"],
            "glasses_confidence": clip_res["confidence"],
            "clip_metadata": json.dumps(clip_res.get("clip_metadata", {})),
            # Retrieve the embedding if present (None if return_embeddings=False)
            "clip_embedding": (
                np.array(clip_res["clip_embedding"], dtype=np.float32).tolist()
                if clip_res.get("clip_embedding") is not None
                else None
            ),
        }
        final_rows.append(row)

    return pd.DataFrame(final_rows, columns=columns)
