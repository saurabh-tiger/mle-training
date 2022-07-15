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
conda env create -f deploy/conda/env.yml

# Create wheel or tar file packge
python3 setup.py sdist bdist_wheel 

python3 setup.py install

# Run project
python3 src/main.py
```

## To create package
```bash
pip install -e .
```

## Things I did and need to verfiy
- [x] Commented out visualization and correlation
- [x] Didn't see the use of compare_props, so commented it as well.
- [ ] Logger code seems redundant for every module.
- [ ] Need do this ```from src import log_configurar``` instead of ```import log_configurar```.
- [ ] How can I move logging_default_config from src/log_configurar to setup.cfg file   
## Ignore
- [Writing Test Cases for Machine Learning systems.](https://www.analyticsvidhya.com/blog/2022/01/writing-test-cases-for-machine-learning/)
- [Documenting Python code with Sphinx](https://towardsdatascience.com/documenting-python-code-with-sphinx-554e1d6c4f6d)
```ini
- [Day 3- MLOPS End To End Implementation With Deployment- Machine Learning](https://youtu.be/IoAbE4dXb9w)
# Log-configuration.ini
[loggers]
keys=root

[handlers]
keys=consoleHandler, fileHandler

[formatters]
keys=extend

[logger_root]
level=NOTSET
handlers=consoleHandler, fileHandler

[handler_fileHandler]
class=FileHandler
level=NOTSET
formatter=extend
# filename='logs/application.log'
args=('logs/application.log', 'a')

[handler_consoleHandler]
class=StreamHandler
level=NOTSET
formatter=extend
args=(sys.stdout,)

[formatter_extend]
format=%(asctime)s [%(levelname)s] %(filename)s - %(funcName)s - %(message)s
datefmt=%Y-%m-%d %H:%M:%S
validate=True
```

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "houseValuePredictionZ"
version = "0.0.1"
authors = [
  { name="Saurabh Zinjad", email="saurabh.zinjad@tigeranalytics.com" },
]
description = ""House Value Prediction""
readme = "README.md"
license = { file="LICENSE" }
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]

[project.urls]
"Homepage" = "https://github.com/saurabh-tiger/mle-training"
"Bug Tracker" = "https://github.com/saurabh-tiger/mle-training/issues"
```