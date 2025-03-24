from pathlib import Path

import pyarrow as pa

# Automatically determine project root (assuming config.py is in image_filter_pipeline/config/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # One level up from 'config'

DATA_DIR = PROJECT_ROOT / "data"

PATHS = {
    "RAW_DATA_DIR": DATA_DIR / "raw",
    "INTERIM_DATA_DIR": DATA_DIR / "interim",
    "PROCESSED_DATA_DIR": DATA_DIR / "processed",
}

HUGGINGFACE_CONFIG = {
    "DATASET": "wikimedia/wit_base",
    "DATA_FILES": [
        "hf://datasets/wikimedia/wit_base/data/train-00000-of-00330.parquet",
        "hf://datasets/wikimedia/wit_base/data/train-00001-of-00330.parquet",
    ],
}

META_INFER = {
    "image_url": str,
    "face_confidence": float,
    "bbox": str,
    "glasses_label": str,
    "glasses_confidence": float,
    "clip_metadata": str,
    "clip_embedding": object,
}

PARQUET_SCHEMA = pa.schema(
    [
        pa.field("image_url", pa.string()),
        pa.field("face_confidence", pa.float64()),
        pa.field("bbox", pa.string()),
        pa.field("glasses_label", pa.string()),
        pa.field("glasses_confidence", pa.float64()),
        pa.field("clip_metadata", pa.string()),
        pa.field("clip_embedding", pa.list_(pa.float32())),
    ]
)

PARQUET_CONFIG = {
    "engine": "pyarrow",
    "compression": "snappy",
    "write_metadata_file": False,
    "overwrite": True,
    "npartitions": 2,
}

PARQUET_FILES = [
    PATHS["RAW_DATA_DIR"] / "train-00000-of-00330.parquet",
    PATHS["RAW_DATA_DIR"] / "train-00001-of-00330.parquet",
]
