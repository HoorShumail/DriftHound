"""Small utilities for validating and storing tabular data."""

from pathlib import Path

import pandas as pd


def validate_schema(df: pd.DataFrame, expected_columns) -> bool:
    """Validate that required columns exist and contain at least one value."""
    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    null_columns = [column for column in df.columns if df[column].isna().all()]
    if null_columns:
        raise ValueError(f"Columns are entirely null: {null_columns}")
    return True


def ingest_data(source_df: pd.DataFrame, output_path, expected_columns=None) -> str:
    """Validate, deduplicate, and write a dataframe to a parquet file."""
    if expected_columns is not None:
        validate_schema(source_df, expected_columns)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_df.drop_duplicates().to_parquet(output, index=False)
    return str(output)
