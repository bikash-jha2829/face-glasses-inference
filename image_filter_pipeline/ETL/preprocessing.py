import ast
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def extract_bytes(x):
    try:
        if isinstance(x, dict):
            return x.get("bytes", None)
        elif isinstance(x, str):
            # Try to convert string to dict using ast.literal_eval
            d = ast.literal_eval(x)
            if isinstance(d, dict):
                return d.get("bytes", None)
    except Exception as e:
        print(f"record: {x} , {e}")
        return None


def preprocess_partition(df_partition):
    """
    For each partition:
      - Uses the already extracted 'image_bytes'
      - Preprocesses these bytes to create a 'processed_image' column.
    """
    meta = df_partition._meta.copy()
    meta["image_bytes"] = pd.Series(dtype="object")
    try:
        df_small = df_partition.map_partitions(
            lambda df: df.assign(image_bytes=df["image"].map(extract_bytes)), meta=meta
        )
    except Exception as e:
        logger.exception("Error in partition preprocessing.")
        raise e
    return df_small[
        (df_small["original_height"] >= 100) & (df_small["original_width"] >= 100)
    ][["image_bytes", "image_url"]]
