import pandas as pd
import pytest

from src.ingest_data import data_preprocessing, load_housing_data


def test_load_housing_data():
    func_generated_df = load_housing_data()
    csv_file_df = pd.read_csv("./data/raw/housing.csv")
    assert func_generated_df.shape == csv_file_df.shape


def test_data_preprocessing():
    func_generated_df = data_preprocessing()
    train_csv_df = pd.read_csv("data/processed/train_data.csv")
    test_csv_df = pd.read_csv("data/processed/test_data.csv")
    csv_df = pd.concat([train_csv_df, test_csv_df])

    assert func_generated_df.shape == csv_df.shape
