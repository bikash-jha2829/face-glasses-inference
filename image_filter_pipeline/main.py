import logging
import time

from dask.distributed import progress

from ETL import ingestion, preprocessing
from ETL.storage import store_parquet
from config.config import PARQUET_FILES, META_INFER
from config.settings import DASK_DF_PROCESSING_CONFIG
from models.model_pipeline import run_inference_partition
from ray_dask_init import initialize_ray_and_dask

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    """Main function to load, process, and run inference on the dataset using Dask and Ray."""

    # Initialize Ray and Dask (using the distributed scheduler)
    initialize_ray_and_dask()

    # Load data from Parquet files
    logger.info("Loading data...")
    df = ingestion.load_data(PARQUET_FILES)

    # Extract image bytes and apply preprocessing
    logger.info("Extracting image bytes and preprocessing...")
    df_filtered = preprocessing.preprocess_partition(df)

    # Repartition the DataFrame to optimize parallel execution
    npartitions = DASK_DF_PROCESSING_CONFIG.get("npartitions_for_inference", 32)
    logger.info(f"Repartitioning DataFrame into {npartitions} partitions...")

    start_time = time.time()
    df_filtered = df_filtered.repartition(npartitions=npartitions)
    logger.info(f"Repartitioning completed in {time.time() - start_time:.2f} seconds.")

    # Apply inference function to each partition
    batch_size = DASK_DF_PROCESSING_CONFIG.get("inference_batch_size", 64)
    logger.info(f"Running inference with batch size {batch_size}...")

    start_time = time.time()
    df_results = df_filtered.map_partitions(
        run_inference_partition,
        return_embeddings=True,
        batch_size=batch_size,
        meta=META_INFER,
    )

    # Comment the line below if you dont have enough memory to persist the dataset
    df_results = df_results.persist()

    # Display progress while computations are running
    progress(df_results)
    #
    # # Compute the final result and log execution time
    # result_df = df_results.compute()

    store_parquet(
        df_results,
        "/Users/bikash/stability-ai/version1/image_filter_pipeline/data/processed",
    )
    logger.info(f"Inference completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
