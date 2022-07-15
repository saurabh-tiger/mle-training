""" Script for data collection, ingestion and preprocessing."""
import argparse
import configparser
import os
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# Import custom logger
from src import log_configurar

# Read configuration
config = configparser.ConfigParser()
config.read("setup.cfg")


def get_args():
    """Parse command line arugments.

    Parameters
    ----------
    None

    Returns
    -------
    <class 'argparse.Namespace'>
        parsed arguments
    """
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument(
        "-op", "--output-path", help="Output folder to store downloaded data.", default=os.path.join("data")
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Mention priority of logs according to severity",
    )
    parser.add_argument(
        "-lp", "--log-path", help="Path where logs while get store", default=None,
    )
    parser.add_argument(
        "-ncl", "--no-console-log", action="store_false", help=" whether or not to write logs to the console"
    )

    # parse arugments
    try:
        return parser.parse_args()
    except SystemExit as se:
        print(f"Unable to parse default argparse arguments. It may only occur for pytest. Here is error: {se}")
        logger.warning(f"Unable to parse default argparse arguments. It may only occur for pytest. Here is error: {se}")
        return parser.parse_args([])


args = get_args()
output_path = args.output_path if args.output_path else os.path.join("data")

# Configure logger
logger = log_configurar.configure_logger(log_file=args.log_path, console=args.no_console_log, log_level=args.log_level,)

HOUSING_URL = config["data"]["housing-url"]
HOUSING_PATH = os.path.join(output_path, "raw")


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Fetch housing data.

    Download dataset from given URL. and then store it in given path.

    Parameters
    ----------
    housing_url:string
        Dataset URL
    housing_path: string
        Path to store the new data

    Returns
    ------
        None    
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)  # noqa
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    """Load housing data in Pandas Dataframe.

    Parameters
    ----------
    housing_path: string
        path where data is stored

    Returns
    -------
    Pandas.DataFrame
        dataframe loaded with data from csv file
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def data_preprocessing():
    """Run data preprocessing step.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.dataframe
        Preprocessed dataframe
    """

    # Fetch and Load data
    logger.debug("Fetching and loading dataset")
    fetch_housing_data()
    housing_data = load_housing_data()

    # divide data into target and label category
    housing_targets = housing_data.drop("median_house_value", axis=1)
    housing_labels = housing_data["median_house_value"].copy()

    # handling missing data using imputation
    logger.debug("Handling missing values.")
    imputer = SimpleImputer(strategy="median")
    housing_subset_data = housing_targets.drop("ocean_proximity", axis=1)
    imputer.fit(housing_subset_data)
    X = imputer.transform(housing_subset_data)

    # data preprocessing step
    logger.debug("Preprocessing the dataset.")
    housing_not_null = pd.DataFrame(X, columns=housing_subset_data.columns, index=housing_targets.index)
    housing_not_null["rooms_per_household"] = housing_not_null["total_rooms"] / housing_not_null["households"]
    housing_not_null["bedrooms_per_room"] = housing_not_null["total_bedrooms"] / housing_not_null["total_rooms"]
    housing_not_null["population_per_household"] = housing_not_null["population"] / housing_not_null["households"]
    housing_processed = housing_not_null.join(pd.get_dummies(housing_targets[["ocean_proximity"]], drop_first=True))
    housing_processed = housing_processed.join(housing_labels)

    return housing_processed


def save_train_test_data(
    train_data, test_data, store_path, train_csv_name="train_data.csv", test_csv_name="test_data.csv"
):
    """Save given pandas.dataframe in given storage path with file_name

    Parameters
    ----------
    train_data: pandas.dataframe
        preprocessed train dataframe to store.
    test_data: pandas.dataframe
        preprocessed test dataframe to store.
    store_path: string
        path where dataframe will get stored.
    train_csv_name: string
        name which will get assigned to stored train file.
    test_csv_name: string
        name which will get assigned to stored test file.

    Returns
    -------
    None
    """
    train_data.to_csv(os.path.join(store_path, train_csv_name), index=False)
    test_data.to_csv(os.path.join(store_path, test_csv_name), index=False)
    return


def split_train_test_data():
    """Split preprocessed dataset into train-test and target-label section.

    Parameters
    ----------
    None

    Returns
    -------
    pandas.dataframe
        training dataset
    pandas.dataframe 
        testing dataset
    """
    processed_data = data_preprocessing()

    # Split dataset into train and test using stratifed shuffle technique
    logger.debug("Spliting dataset into train and test set using stratifed shuffle technique.")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    processed_data["income_cat"] = pd.cut(
        processed_data["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5],
    )
    for train_index, test_index in split.split(processed_data, processed_data["income_cat"]):
        train_set = processed_data.loc[train_index]
        test_set = processed_data.loc[test_index]

    for set_ in (train_set, test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    # storing processed dataset
    store_path = os.path.join(output_path, "processed")
    logger.info(f'storing train and test processed dataset into path "{store_path}"')
    save_train_test_data(train_data=train_set, test_data=test_set, store_path=store_path)
    logger.debug("End of data collection and preprocessing step.")
    return train_set, test_set


def data_collection_preprocessing():
    """Run ingest_data python script for all data preprocessing step."""
    return split_train_test_data()


if __name__ == "__main__":
    logger.debug("Start Data Collection & Preprocessing Phase =======")
    data_collection_preprocessing()
    logger.debug("End Data Collection & Preprocessing Phase =======")
