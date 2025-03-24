# tests/conftest.py
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def test_parquet_path():
    """
    Returns the absolute path to a small test Parquet file.
    This file should have the same schema as your production data.
    """
    return "/Users/bikash/stability-ai/version1/image_filter_pipeline/tests/data/testdata.parquet"


@pytest.fixture(scope="session")
def df_expected():
    """
    Loads the expected results from a CSV file.
    The CSV contains the following columns:
      image_url, face_confidence, bbox, glasses_label, glasses_confidence, clip_metadata, clip_embedding
    We use 'glasses_label' and 'glasses_confidence' as the ground truth.
    """
    return pd.read_csv(
        "/Users/bikash/stability-ai/version1/image_filter_pipeline/tests/data/expected_results.csv"
    )


@pytest.fixture
def test_face_image():
    """Return a PIL Image that is known to contain a face."""
    return Image.open(
        "/Users/bikash/stability-ai/version1/image_filter_pipeline/tests/data/images/face.png"
    )


@pytest.fixture
def test_no_face_image():
    """Return a PIL Image that is known to contain no faces."""
    return Image.open(
        "/Users/bikash/stability-ai/version1/image_filter_pipeline/tests/data/images/noface.png"
    )


@pytest.fixture
def test_clip_face_image():
    """
    Fixture that loads a test face image for the CLIP classifier.
    Make sure the image file exists at the specified path.
    """
    return Image.open(
        "/Users/bikash/stability-ai/version1/image_filter_pipeline/tests/data/images/face.png"
    )
