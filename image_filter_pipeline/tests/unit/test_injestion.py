from ETL.ingestion import load_data


def test_load_data_valid_entries(test_parquet_path):
    """
    Test that the loaded Parquet file contains valid entries.
    Ensures that all columns contain valid data.
    """
    df = load_data([test_parquet_path])

    # Compute Dask DataFrame to Pandas for validation
    pandas_df = df.compute()

    # Ensure dataframe is not empty
    assert not pandas_df.empty, "Loaded DataFrame is empty, expected valid data."

    # Validate 'image_url' column (should be non-null and a valid string)
    assert pandas_df["image_url"].notna().all(), "image_url column contains NaN values."
    assert (
        pandas_df["image_url"]
        .apply(lambda x: isinstance(x, str) and x.startswith("http"))
        .all()
    ), "image_url column contains invalid URLs."

    # Validate 'image' column (should not be null)
    assert pandas_df["image"].notna().all(), "image column contains NaN values."

    # Validate 'original_height' & 'original_width' (should be positive integers)
    assert (
        pandas_df["original_height"].notna().all()
    ), "original_height column contains NaN values."
    assert (
        pandas_df["original_width"].notna().all()
    ), "original_width column contains NaN values."
    assert (
        pandas_df["original_height"] > 0
    ).all(), "original_height contains non-positive values."
    assert (
        pandas_df["original_width"] > 0
    ).all(), "original_width contains non-positive values."
