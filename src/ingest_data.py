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
import log_configurar

# Read configuration
config = configparser.ConfigParser()
config.read("setup.cfg")

# Parse arugments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-op", "--output_path", help="Output folder to store downloaded data.", default=os.path.join("data")
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
args = parser.parse_args()

# Configure logger
logger = log_configurar.configure_logger(log_file=args.log_path, console=args.no_console_log, log_level=args.log_level,)

HOUSING_URL = config["data"]["housing-url"]
HOUSING_PATH = os.path.join(args.output_path, "raw")


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """Fetch housing data.

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
        housing_path: string)
            path where data is stored

    Returns
    -------
        (Pandas.DataFrame)
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def income_cat_proportions(data):
    """Calculate proportion of each unique value."""
    return data["income_cat"].value_counts() / len(data)


def data_collection_preprocessing():
    """Run data collection, Ingestion and preprocessing step."""

    # Fetch and Load data
    logger.debug("Fetching and loading dataset.")
    fetch_housing_data()
    housing = load_housing_data()

    # data preprocessing
    logger.debug("Preprocessing the dataset.")
    housing["income_cat"] = pd.cut(
        housing["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    logger.debug("Dividing dataset into train and test datasets.")
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    housing = strat_train_set.copy()

    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing = strat_train_set.drop("median_house_value", axis=1)  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
    housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    housing_tr["population_per_household"] = housing_tr["population"] / housing_tr["households"]

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))
    housing_prepared = housing_prepared.join(strat_train_set["median_house_value"].copy())

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(X_test_prepared, columns=X_test_num.columns, index=X_test.index)
    X_test_prepared["rooms_per_household"] = X_test_prepared["total_rooms"] / X_test_prepared["households"]
    X_test_prepared["bedrooms_per_room"] = X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    X_test_prepared["population_per_household"] = X_test_prepared["population"] / X_test_prepared["households"]

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))
    X_test_prepared = X_test_prepared.join(strat_test_set["median_house_value"].copy())

    store_path = os.path.join(args.output_path, "processed")
    os.makedirs(store_path, exist_ok=True)

    logger.info(f'storing train and test processed dataset into path "{store_path}"')
    housing_prepared.to_csv(os.path.join(store_path, "train.csv"), index=False)
    X_test_prepared.to_csv(os.path.join(store_path, "test.csv"), index=False)
    logger.debug("End of data collection and preprocessing step.")


if __name__ == "__main__":
    logger.debug("Start Data Collection & Preprocessing Phase =======")
    data_collection_preprocessing()
    logger.debug("End Data Collection & Preprocessing Phase =======")
