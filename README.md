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

## Setup Conda environment
```bash
conda env create -f env.yml
```

## To excute the script
```bash
python house-value-prediction.py
```


## Things I did and need to verfiy
- [x] Commented out visualization and correlation
- [x] Didn't see the use of compare_props, so commented it as well.
- [ ] Logger code seems redundant for every module.

## Ignore
```ini
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