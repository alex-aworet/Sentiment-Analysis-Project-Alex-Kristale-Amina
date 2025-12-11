import os
import sys
import tempfile
import pandas as pd
import pytest

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    )

from src.data_extraction import load_file, check_columns  # noqa: E402

columns = ["content", "score",
           "reviewId", "userName",
           "userImage", "thumbsUpCount",
           "reviewCreatedVersion", "at",
           "replyContent", "repliedAt",
           "sortOrder", "appId"]


def make_valid_df():
    """
    Create a valid dataframe that contains all the required columns
    each column has a single value 'ok' for simplicity
    """
    data = {col: ["ok"] for col in columns}
    return pd.DataFrame(data)


def temp_csv_file(df):
    """
    Create a temp csv file from the given dataframe
    returns the path to this temp file
    """
    data, path = tempfile.mkstemp(suffix='.csv')
    os.close(data)
    df.to_csv(path, index=False)
    return path


def test_load_file_valid():
    """
    Test that load_file correctly loads a valid csv file
    the returned object is a dataframe
    data matches the input dataframe
    all expected columns are present
    """
    df = make_valid_df()
    path = temp_csv_file(df)
    loaded_df = load_file(path)

    assert loaded_df.equals(df)
    assert loaded_df is not None
    assert isinstance(loaded_df, pd.DataFrame)
    assert not loaded_df.empty
    assert list(loaded_df.columns) == list(df.columns)
    assert loaded_df.iloc[0]["content"] == "ok"
    assert "replyContent" in loaded_df.columns


def test_wrong_file_extension():
    """
    Test that load_file raises a value error if the file
    does not have .csv extension
    """
    data, path = tempfile.mkstemp(suffix='.txt')
    os.close(data)
    try:
        with pytest.raises(ValueError) as e:
            load_file(path)
        assert "csv" in str(e).lower()
        assert "must" in str(e).lower()
    finally:
        os.remove(path)


def test_missing_file():
    """
    load_file returns None if the file does not exist
    """
    path = "non_existent_file.csv"
    loaded_df = load_file(path)

    assert loaded_df is None


def test_check_columns_valid():
    """
    Test -> check_columns returns True for a dataframe
    with all required columns
    """
    df = make_valid_df()
    assert isinstance(check_columns(df), bool)
    assert len(df.columns) == len(columns)
    for col in df.columns:
        assert col in df.columns


def test_check_columns_missing():
    """
    Test -> check_columns returns False for a dataframe
    missing one required column
    """
    df = make_valid_df()
    df = df.drop(columns=["score"])
    assert check_columns(df) is False
    assert "score" not in df.columns
    assert len(df.columns) == len(columns) - 1


def test_check_columns_more_missing_columns():
    """
    Test -> check_columns returns False for a dataframe
    missing multiple required columns
    """
    df = make_valid_df()
    df = df.drop(columns=["score", "userName", "appId"])
    assert check_columns(df) is False
    assert "score" not in df.columns
    assert "userName" not in df.columns
    assert "appId" not in df.columns
    assert len(df.columns) == len(columns) - 3


def test_end_to_end():
    """
    For the both load_file and check_columns
    -> load a valid csv file and check that all required columns are present
    -> Remove a column and verify that validation fails
    """
    df = make_valid_df()
    path = temp_csv_file(df)
    loaded_df = load_file(path)
    # All columns present
    assert loaded_df is not None
    assert check_columns(loaded_df) is True
    # one column removed
    loaded_df = loaded_df.drop(columns=["at"])
    assert check_columns(loaded_df) is False

