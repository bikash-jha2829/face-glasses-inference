# image_filter_pipeline/storage.py
import logging

import dask.dataframe as dd

from config import config  # Import config module

logger = logging.getLogger(__name__)


def store_parquet(df: dd.DataFrame, output_path: str = None) -> None:
    """
    Store selected metadata (e.g., image_url, face_detections, and clip_embedding) as a Parquet file.

    Parameters:
        df (dd.DataFrame): The Dask DataFrame to store.
        output_path (str): The output path for storing the Parquet file.
    """

    def with_snappy(n):
        return f"part-{n}.snappy.parquet"

    # Fetch Parquet settings from config
    parquet_schema = config.PARQUET_SCHEMA
    parquet_options = config.PARQUET_CONFIG

    # Ensure the DataFrame is partitioned correctly
    npartitions = parquet_options.get("npartitions", 2)
    df_results2 = df.repartition(npartitions=npartitions)

    # Store as Parquet
    logger.info(f"Saving DataFrame to {output_path} with {npartitions} files.")

    df_results2.to_parquet(
        output_path,
        engine=parquet_options.get("engine", "pyarrow"),
        compression=parquet_options.get("compression", "snappy"),
        write_metadata_file=parquet_options.get("write_metadata_file", False),
        name_function=with_snappy,
        schema=parquet_schema,
    )

    logger.info(f"Successfully stored Parquet file at {output_path}.")
