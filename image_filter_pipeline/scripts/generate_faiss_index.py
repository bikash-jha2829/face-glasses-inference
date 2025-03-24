"""
generate_faiss_index.py

This script leverages Dask and Ray to efficiently generate a FAISS index
from image embedding data stored in Parquet files. It:
  - Loads the Parquet files from 'data/processed' (relative to the project root)
  - Extracts the 'clip_embedding' from each row in parallel using Ray
  - Combines the embeddings and builds a FAISS IndexFlatL2 index for vector similarity search
  - Saves the resulting FAISS index to disk

Usage:
    python generate_faiss_index.py
"""

import os
import numpy as np
import dask.dataframe as dd
import ray
import faiss
from dask import delayed


# Remote function to process a single partition of the dataframe.
@ray.remote
def process_partition(partition_df):
    """
    Process a single partition (pandas DataFrame) to extract embeddings.
    Assumes that 'clip_embedding' column contains a list-like structure of floats.

    Returns:
        np.ndarray: A 2D numpy array of embeddings with shape (n_vectors, vector_dim)
    """
    # Convert the column of embeddings to a list of numpy arrays.
    # We assume each 'clip_embedding' is already a list of floats.
    embeddings_list = partition_df['clip_embedding'].tolist()

    # Convert to a 2D numpy array (stacking all embeddings vertically)
    return np.vstack(embeddings_list)


def main():
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    print("Ray initialized.")

    # Construct the path to the processed data directory.
    # Adjust the path if needed. Here we assume that the script is run from the project root.
    data_dir = os.path.join(os.getcwd(), "data", "processed", "*.parquet")
    print("Reading Parquet files from:", data_dir)

    # Load all parquet files using a glob pattern with Dask.
    df = dd.read_parquet(data_dir, engine="pyarrow")
    print("Dask DataFrame loaded with {} partitions.".format(df.npartitions))

    # Optionally repartition for optimal parallelism (depends on your dataset size).
    # For example, to have 10 partitions:
    # df = df.repartition(npartitions=10)

    # Convert the Dask DataFrame into a list of delayed pandas DataFrames (one per partition).
    delayed_partitions = df.to_delayed()

    # Launch Ray tasks to process each partition.
    futures = [process_partition.remote(part.compute()) for part in delayed_partitions]

    # Gather the results from all partitions.
    embedding_arrays = ray.get(futures)

    # Combine all partition embeddings into one large numpy array.
    embeddings = np.vstack(embedding_arrays)
    print("Combined embeddings shape:", embeddings.shape)

    # Build a FAISS index.
    # We'll use IndexFlatL2, which performs L2 (Euclidean) distance searches.
    vector_dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(vector_dim)

    # Add all embeddings to the FAISS index.
    index.add(embeddings)
    print("FAISS index built with {} vectors.".format(index.ntotal))

    # Save the FAISS index to disk.
    index_filename = "faiss_index.index"
    faiss.write_index(index, index_filename)
    print("FAISS index saved to:", index_filename)

    # Shutdown Ray.
    ray.shutdown()
    print("Ray shutdown completed.")


if __name__ == "__main__":
    main()
