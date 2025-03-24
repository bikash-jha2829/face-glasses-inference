# Steps to Run the Project

## Overview

This guide will walk you through setting up and running the project.

## Developer Notes

## PS

- 🚗 **Pre-Downloaded Files:** All Parquet files have been pre-downloaded and are stored in the `data/raw` directory to save time on iterations. If a file is missing, the script will automatically download it (leverage CACHE after first time).
- ☁️ **Production Scenario:** In a real production environment, raw files would be stored in an S3/GCS bucket, and processed data would be pushed to an archive—keeping the raw data intact rather than fetching from Hugging Face every time.
- 💻 **Hardware Setup:** I ran the pipeline on a Mac M1 using minimal memory settings (see the Memory section below).
- 📁 **Output Location:** The processed Parquet files are stored in the `data/processed` directory. There’s also a script (separate from the main code) to push data to a Hugging Face dataset, which can be integrated if needed.
- 🔍 **FAISS & Vector Databases (In progress):** (refer `scripts/generate_faiss_index.py & notebooks/faiss`   I've done some basic work on creating a FAISS index and exploring vector databases. I'd love to dive deeper if time allowed.
- ⏱️ **Execution Time:** Running the pipeline with the pre-downloaded files takes approximately 4.3 minutes.
- 📊 **Monitoring:** Check out our Dask dashboard for real-time insights on CPU and memory utilization, as well as stage progress.
- 🚀 **Ray Integration:** Ray is incorporated for tasks (primarily in the FAISS script) where Dask’s parallelization isn’t enough, while Dask handles the bulk of the processing.
- 🔍  **Sample search**: refer notebooks containing example of search using dask and dask sql


## Prerequisites

-[ ] **Python "^3.10"**: Ensure that Python is installed on your machine. 
-[ ] **Poetry**: This project uses [Poetry](https://python-poetry.org/) for dependency management. 


## Setup Instructions 

1. **Install Poetry**

   If you don't have Poetry installed, you can install it using pip:

   ```bash
   pip install poetry
    ```

   ```bash
   poetry install
   

## 2. Steps to Run the Project
### Prerequisites

Before running the pipeline, ensure you have the following installed:
1. [ ] Poetry (for dependency management)
2. [ ] Python "^3.10" (Used: Python 3.10.6) # can leverage pyenv

The project’s configuration is centralized in two key files: `config/config.py` and `config/settings.py`. These files are crucial for defining settings like file paths, batch sizes, and other environment-specific variables.

```html
## Configuration Overview

- **config/config.py**  
  - Sets project root and data directories (raw, interim, processed).  
  - Defines Parquet file paths and schema with PyArrow.

- **config/settings.py**  
  - Configures Ray (CPU count, spill directory, object store memory).  
  - Sets Dask worker parameters (memory limits, worker count) and model/inference settings.

These files centralize environment-specific settings (e.g., file paths, batch sizes) for easy customization and consistent project behavior.

```


### 1. Running the Pipeline Using Makefile
Makefile simplifies execution by managing dependencies and execution order.

## 1. Running the Pipeline Using Makefile
Makefile simplifies execution by managing dependencies and execution order.

### 1.1 make help : 
```bash
 make help
====================================================
                Available Make Targets
====================================================
make setup               - Install Poetry dependencies
make download-data       - Download raw data files via script
make run-pipeline        - Run the image filter pipeline (Ray + Dask)
make clean-data          - Clean processed and interim data folders
make test                - Run unit tests (pytest)
make format              - Auto-format code with Black
make ci-format           - Check code format (CI/CD)
====================================================

```
### 1.2 Run  Pipeline 

```bash
make run-pipeline 
```
<details> <summary><b>Execution Log (Click to Expand)</b></summary>

```angular2html

make run-pipeline 
====================================================
Running image filter pipeline with Dask & Ray...
====================================================
python main.py
2025-03-24 12:34:46,268 INFO worker.py:1774 -- Started a local Ray instance. View the dashboard at http://127.0.0.1:8266 
Dask client initialized. Dashboard available at: http://127.0.0.1:55157/status
INFO:__main__:Loading data...
Checking Parquet Files:   0%|                                                                                                                                                                                                                                                                | 0/2 [00:00<?, ?it/s]INFO:ETL.ingestion:File /Users/bikash/stability-ai/version1/image_filter_pipeline/data/raw/train-00000-of-00330.parquet not found. Downloading from Hugging Face...
INFO:ETL.ingestion:Downloaded /Users/bikash/stability-ai/version1/image_filter_pipeline/data/raw/train-00000-of-00330.parquet from wikimedia/wit_base/data/train-00000-of-00330.parquet
Checking Parquet Files:  50%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                            | 1/2 [00:00<00:00,  1.58it/s]INFO:ETL.ingestion:File /Users/bikash/stability-ai/version1/image_filter_pipeline/data/raw/train-00001-of-00330.parquet not found. Downloading from Hugging Face...
INFO:ETL.ingestion:Downloaded /Users/bikash/stability-ai/version1/image_filter_pipeline/data/raw/train-00001-of-00330.parquet from wikimedia/wit_base/data/train-00001-of-00330.parquet
Checking Parquet Files: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.74it/s]
INFO:ETL.ingestion:Loaded data from [PosixPath('/Users/bikash/stability-ai/version1/image_filter_pipeline/data/raw/train-00000-of-00330.parquet'), PosixPath('/Users/bikash/stability-ai/version1/image_filter_pipeline/data/raw/train-00001-of-00330.parquet')] with 4 partitions
INFO:__main__:Extracting image bytes and preprocessing...
INFO:__main__:Repartitioning DataFrame into 32 partitions...
INFO:__main__:Repartitioning completed in 0.00 seconds.
INFO:__main__:Running inference with batch size 64...
[####################                    ] | 51% Completed | 51.7s
.....

INFO:ETL.storage:Saving DataFrame to /Users/bikash/stability-ai/version1/image_filter_pipeline/data/processed with 2 files.
INFO:ETL.storage:Successfully stored Parquet file at /Users/bikash/stability-ai/version1/image_filter_pipeline/data/processed.
INFO:__main__:Inference completed in 248.33 seconds.
✅ Pipeline execution completed!
```
</details>


### 1.6 Last for not the least Run  Tests:
```bash
make test
```

<details>
  <summary><b>Test log(click to open)</b></summary>

```bash
 make test        
====================================================
Running unit tests with pytest...
====================================================
PYTHONPATH=. pytest tests/
============================= test session starts ================================
platform darwin -- Python 3.11.9, pytest-8.3.5, pluggy-1.5.0
rootdir: /Users/bikash/stability-ai/version1/image_filter_pipeline
configfile: pyproject.toml
plugins: anyio-4.8.0
collected 8 items                                                                                                                                                                                                                                                                                                 

tests/integration/test_end2end.py .                                                                                                                                                                                                                                                                         [ 12%]
tests/unit/test_clip_classifier.py ...                                                                                                                                                                                                                                                                      [ 50%]
tests/unit/test_injestion.py .                                                                                                                                                                                                                                                                              [ 62%]
tests/unit/test_yolo_classifer.py ...                                                                                                                                                                                                                                                                       [100%]

================================= 8 passed in 12.04s =================================
✅ Tests completed!

```
</details>



### 1.3 **Format Code**
```bash
make format
```


## **Running the Pipeline Locally (my favorite)**
If you prefer to manually start components without using Makefile, follow these steps:

```bash
cd image_filter_pipeline
python main.py
```

### What happens after running the project?
After running the project, the pipeline processes the the image data, generates Parquet files along-with embeddings

Once it is produced you can produce FAISS Index as well leverage notebook code and scripts
Researcher can leverage Parquet file to filter data (refer `notebooks/search_query_dask_sql.ipynb`)




