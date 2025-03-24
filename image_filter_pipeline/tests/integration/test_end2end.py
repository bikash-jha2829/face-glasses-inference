# tests/test_end_to_end.py
import math

from ETL.ingestion import load_data
from ETL.preprocessing import preprocess_partition
from models.model_pipeline import run_inference_partition


def test_end_to_end_pipeline(test_parquet_path, df_expected):
    """
    End-to-end test:
      - Load a small test Parquet input.
      - Run ingestion, preprocessing, and inference.
      - Compare the resulting DataFrame against the expected CSV.
    """
    # 1) Ingestion: load the small test Parquet file.
    df_loaded = load_data([test_parquet_path])  # Returns a Dask DataFrame.

    # 2) Preprocess: extract image bytes, filter by size, etc.
    df_preprocessed = preprocess_partition(df_loaded)
    pdf_preprocessed = df_preprocessed.compute()  # Convert to a Pandas DataFrame.

    # 3) Inference: run inference on the preprocessed data.
    df_result = run_inference_partition(
        df_partition=pdf_preprocessed,
        batch_size=4,  # Small batch size for test.
        return_embeddings=False,  # Omit embeddings for this test.
    )

    # 4) Build a mapping from image_url to expected results.
    # Use 'glasses_label' and 'glasses_confidence' from the CSV as expected values.
    expected_map = {}
    for _, row in df_expected.iterrows():
        expected_map[row["image_url"]] = {
            "expected_label": row["glasses_label"],
            "expected_conf": float(row["glasses_confidence"]),
        }

    # 5) Validate each result row.
    for _, row in df_result.iterrows():
        url = row["image_url"]
        if url not in expected_map:
            # If the image is not in our expected map, we skip it.
            continue

        predicted_label = row["glasses_label"]
        predicted_conf = row["glasses_confidence"]
        expected_label = expected_map[url]["expected_label"]
        expected_conf = expected_map[url]["expected_conf"]

        # Check that the predicted label exactly matches the expected label.
        assert (
            predicted_label == expected_label
        ), f"For image {url}, got label '{predicted_label}' but expected '{expected_label}'."

        # Check that the predicted confidence is close enough to the expected confidence.
        tolerance = 0.05
        assert math.isclose(
            predicted_conf, expected_conf, abs_tol=tolerance
        ), f"For image {url}, confidence {predicted_conf} is not within ±{tolerance} of expected {expected_conf}."
