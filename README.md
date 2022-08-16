# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## Setup and Installation
```bash
# install dependencies for project
$ conda env create -f deploy/conda/env.yml

# Create wheel or tar file packge
$ python3 setup.py sdist bdist_wheel

$ pip install houseValuePrediction-0.0.2-py3-none-any.whl

$ python3 setup.py install
# Or
$ pip install -e .

# Run project
$ python3 src/main.py
```
### NOTE: 
1. [environment yaml](deploy/conda/env.yml) file is stored under deploy/conda directory. click here to navigate.
2. you can find Wheel (`.whl`) and sdist (`tar.gz`) file under [dist](dist/) folder

## Run Test Script 
- To run default test case run following command:
```bash
$ pytest
```

- Go to [`tests`](/tests) directory to se particular test script for specific purpose.
```
$ pytest tests/functional_tests/test_general.py
$ pytest tests/unit_test/test_ingest_data.py
```

## Log Files
All generated log files are stored under [`logs`](/logs/) directory.

## MLFlow Server Launch
```bash
$ mlflow server --backend-store-uri artifacts/mlruns/ --default-artifact-root artifacts/mlruns/ --host 0.0.0.0 --port 5000
```

## Documentation Generation
```bash
$ cd docs/
$ make clean html
```