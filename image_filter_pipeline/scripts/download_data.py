import logging
import shutil
from pathlib import Path
from config.config import HUGGINGFACE_CONFIG, PATHS
from huggingface_hub import hf_hub_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_hugging_face_data_if_missing(local_path: Path, hf_url: str) -> None:
    """
    Check if the local file exists; if not, download it from Hugging Face Hub.
    """
    if not local_path.exists():
        logger.info(f"File {local_path} not found. Downloading from Hugging Face...")
        # Clean hf_url prefix
        hf_url = hf_url.replace("hf://", "").replace("datasets/", "")
        parts = hf_url.split("/")
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])

        downloaded_path = hf_hub_download(
            repo_id=repo_id, filename=filename, repo_type="dataset"
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(downloaded_path, local_path)
        logger.info(f"Downloaded {local_path} from {repo_id}/{filename}")
    else:
        logger.info(f"File {local_path} already exists. Skipping download.")


def download_all_files():
    hf_files = HUGGINGFACE_CONFIG["DATA_FILES"]
    raw_dir = PATHS["RAW_DATA_DIR"]

    for hf_file in hf_files:
        filename = Path(hf_file).name
        local_file_path = raw_dir / filename
        download_hugging_face_data_if_missing(local_file_path, hf_file)


if __name__ == "__main__":
    download_all_files()
