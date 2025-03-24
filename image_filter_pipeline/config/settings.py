# image_filter_pipeline/config/settings.py
import os

from config.config import DATA_DIR

# Ray configuration
RAY_NUM_CPUS = os.cpu_count()  # Number of CPUs to use
RAY_SPILL_DIR = "./spill/"  # Directory for spilling objects
RAY_OBJECT_STORE_MEMORY = 2 * 1024**3  # 2GB for Ray's object store

# Dask configuration
DASK_MEMORY_LIMIT = "8GB"  # Memory limit for each Dask worker
DASK_WORKER_THREADS = 2  # Threads per Dask worker
DASK_NUM_WORKERS = 4  # Number of Dask workers
DASK_SCHEDULER = "ray_dask_get"  # Use Ray's Dask integration scheduler
DASK_SPILL_DIR = "./spill/"  # Spill directory for Dask
DASK_CONFIG = {
    "distributed.worker.memory.target": 0.8,
    "distributed.worker.memory.spill": 0.75,
    "array.chunk-size": "32MB",
    "distributed.comm.timeouts.tcp": "120s",
    "distributed.comm.timeouts.connect": "120s",
    "distributed.worker.heartbeat.interval": "120s",
    "distributed.scheduler.work-stealing": False,
    "logging": {
        "distributed": "ERROR",
        "ray": "ERROR",
        "distributed.shuffle._scheduler_plugin": "ERROR",
    },
}

SUPPRESS_WARNINGS = True

# YOLO (Ultralytics) configuration
YoloConfig = {
    "model_path": DATA_DIR/ "models/yolov11n-face.pt",
    "custom_classes": ["face"],
    "batch_size": 8,
    "img_size": 640,
    "confidence": 0.55,
    "min_face_size": 100,  # Minimum width and height in pixels for valid detection
}
# CLIP Zero-Shot Detection configuration
ClipDetectionConfig = {
    "model_name": "openai/clip-vit-base-patch32",
    "text_descriptions": [
        "A person wearing reading glasses",
        "A person wearing sunglasses",
        "A person wearing no glasses",
    ],
    "threshold": 0.5,  # Similarity threshold for detection
    "batch_size": 8,  # Batch size for CLIP detection
}

DASK_DF_PROCESSING_CONFIG = {
    "inference_batch_size": 64,  # Batch size for inference
    "npartitions_for_inference": 32,  # Number of partitions before inference
    "ingestion_npartitions": 16,
}

# # Storage configuration (Hybrid approach)
# StorageConfig = {
#     "metadata_path": "metadata/",         # Directory to store Parquet file with metadata
#     "faiss_index_path": "faiss_index.index" # File to store FAISS index
# }
