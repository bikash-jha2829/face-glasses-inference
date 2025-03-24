# image_filter_pipeline/models/yolo_detector.py
import logging

import torch
from ultralytics import YOLO

from config.settings import YoloConfig

logger = logging.getLogger(__name__)
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

# Global cache for YOLO detector instance per worker.
_cached_yolo_detector = None

DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


class YOLOFaceDetector:
    def __init__(
        self,
        model_path: str = YoloConfig["model_path"],
        conf_threshold: float = YoloConfig["confidence"],
    ) -> None:
        self.device = DEVICE
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.model.to(self.device)

    def detect_faces_batch(self, images) -> list:
        """
        Detect faces in a batch of images.
        :param images: List of PIL Images or numpy arrays (BGR)
        :return: List (per image) of detection dictionaries with "bbox" and "face_confidence".
        """
        results = self.model.predict(
            images,
            conf=self.conf_threshold,
            device=self.device,
            batch=YoloConfig["batch_size"],
            verbose=False,
            stream=True,
            max_det=2,
        )
        all_detections = []
        for res in results:
            detections = []
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                w, h = x2 - x1, y2 - y1
                # Retrieve confidence score if available.
                conf = (
                    float(box.conf.cpu().numpy()[0]) if hasattr(box, "conf") else None
                )
                if w >= 10 and h >= 10:
                    detections.append(
                        {
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                            "face_confidence": conf,
                        }
                    )
            all_detections.append(detections)
        return all_detections


def get_yolo_detector() -> YOLOFaceDetector:
    """
    Return a cached instance of YOLOFaceDetector per worker.
    """
    global _cached_yolo_detector
    if _cached_yolo_detector is None:
        _cached_yolo_detector = YOLOFaceDetector()
    return _cached_yolo_detector
