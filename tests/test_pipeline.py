import pandas as pd
import pytest

from data.ingestion.pipeline import ingest_data, validate_schema


def test_valid_schema_passes():
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    assert validate_schema(data, ["a", "b"]) is True


def test_missing_column_raises_value_error():
    data = pd.DataFrame({"a": [1, 2]})

    with pytest.raises(ValueError, match="Missing columns"):
        validate_schema(data, ["a", "b"])


def test_all_null_column_raises_value_error():
    data = pd.DataFrame({"a": [1, 2], "b": [None, None]})

    with pytest.raises(ValueError, match="entirely null"):
        validate_schema(data, ["a", "b"])


def test_unexpected_all_null_column_raises_value_error():
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4], "extra": [None, None]})

    with pytest.raises(ValueError, match="entirely null"):
        validate_schema(data, ["a", "b"])


def test_parquet_created_and_readable(tmp_path):
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    output_path = tmp_path / "data.parquet"

    result = ingest_data(data, output_path, expected_columns=["a", "b"])

    assert result == str(output_path)
    assert output_path.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(output_path), data)


def test_duplicates_dropped(tmp_path):
    data = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
    output_path = tmp_path / "deduplicated.parquet"

    ingest_data(data, output_path)

    assert len(pd.read_parquet(output_path)) == 2
