# image_filter_pipeline/ray_dask_init.py
import warnings

import dask
import ray
from dask.distributed import Client
from distributed import LocalCluster
from ray.util.dask import ray_dask_get

from config.settings import (
    SUPPRESS_WARNINGS,
    DASK_CONFIG,
    RAY_OBJECT_STORE_MEMORY,
    RAY_NUM_CPUS,
    DASK_NUM_WORKERS,
    DASK_WORKER_THREADS,
    DASK_MEMORY_LIMIT,
    RAY_SPILL_DIR,
)

# TODO: Create a logger class rather than putting in here
warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
    category=UserWarning,
    module="PIL.Image",
)


def initialize_ray_and_dask():
    """Initialize Ray and configure Dask to use Ray as its scheduler."""
    if SUPPRESS_WARNINGS:
        warnings.filterwarnings("ignore")

    # Set Dask logging configuration
    import logging

    logging.getLogger("distributed").setLevel(DASK_CONFIG["logging"]["distributed"])
    logging.getLogger("ray").setLevel(DASK_CONFIG["logging"]["ray"])

    ray.init(
        ignore_reinit_error=True,
        log_to_driver=False,
        num_cpus=RAY_NUM_CPUS,
        include_dashboard=True,
        object_store_memory=RAY_OBJECT_STORE_MEMORY,
    )

    dask.config.set(scheduler=ray_dask_get)
    dask.config.set(DASK_CONFIG)

    cluster = LocalCluster(
        n_workers=DASK_NUM_WORKERS,
        threads_per_worker=DASK_WORKER_THREADS,
        memory_limit=DASK_MEMORY_LIMIT,
        local_directory=RAY_SPILL_DIR,
    )

    client = Client(cluster)
    print(f"Dask client initialized. Dashboard available at: {client.dashboard_link}")
    return client
