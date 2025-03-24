# image_filter_pipeline/ingestion.py
import logging
import shutil
from pathlib import Path
from typing import List

import dask.dataframe as dd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from config.config import HUGGINGFACE_CONFIG, PARQUET_FILES
from config.settings import DASK_DF_PROCESSING_CONFIG

logger = logging.getLogger(__name__)


def download_file_if_missing(local_path: Path, hf_url: str) -> None:
    """
    Check if the local file exists; if not, download it from Hugging Face Hub.
    """
    if not local_path.exists():
        logger.info(f"File {local_path} not found. Downloading from Hugging Face...")
        # Remove the "hf://" prefix if present
        if hf_url.startswith("hf://"):
            hf_url = hf_url[len("hf://") :]
        if hf_url.startswith("datasets/"):
            hf_url = hf_url[len("datasets/") :]
        parts = hf_url.split("/")
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])

        downloaded_path = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="dataset"
        )
        # Ensure the parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(downloaded_path, local_path)
        logger.info(f"Downloaded {local_path} from {repo_id}/{filename}")
    else:
        logger.info(f"File {local_path} already exists. Skipping download.")


def load_data(parquet_paths: List[str]) -> dd.DataFrame:
    """
    Load Parquet files into a Dask DataFrame.
    For each file in parquet_paths, check if it exists.
    If not, download it using the corresponding URL from HUGGINGFACE_CONFIG.
    """
    # Ensure each parquet file is present, with progress feedback.
    for idx, local_file in enumerate(
        tqdm(parquet_paths, desc="Checking Parquet Files")
    ):
        local_path = Path(local_file)
        try:
            hf_url = HUGGINGFACE_CONFIG["DATA_FILES"][idx]
        except IndexError:
            logger.error("No Hugging Face URL found for file: %s", local_file)
            raise ValueError(
                "Mismatch between local file list and Hugging Face configuration."
            )
        download_file_if_missing(local_path, hf_url)

    try:
        df = dd.read_parquet(
            parquet_paths,
            engine="pyarrow",
            columns=["image", "image_url", "original_height", "original_width"],
        )
        logger.info(
            f"Loaded data from {parquet_paths} with {df.npartitions} partitions"
        )
        df = df.repartition(
            npartitions=DASK_DF_PROCESSING_CONFIG.get("ingestion_npartitions", 16)
        )
        return df
    except Exception as e:
        logger.exception("Failed to load Parquet data.")
        raise e
