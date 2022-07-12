import logging
import logging.config

LOGGING_DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "default_formatter": {
            "format": "%(asctime)s [%(levelname)s] %(filename)s - %(funcName)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "consoleHandler": {
            "level": "NOTSET",
            "formatter": "default_formatter",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "fileHandler": {
            "level": "NOTSET",
            "class": "logging.FileHandler",
            "formatter": "default_formatter",
            "filename": "logs/default.log",
        },
    },
    "loggers": {"": {"level": "NOTSET", "handlers": ["consoleHandler", "fileHandler"]},},
}


def configure_logger(logger=None, cfg=None, log_file=None, console=True, log_level="DEBUG"):
    """Function to setup configurations of logger through function.

    The individual arguments of `log_file`, `console`, `log_level` will overwrite the ones in cfg.

    Parameters
    ----------
        logger:
            Predefined logger object if present. If None a ew logger object will be created from root.
        cfg: dict()
            Configuration of the logging to be implemented by default
        log_file: str
            Path to the log file for logs to be stored
        console: bool
            To include a console handler(logs printing in console)
        log_level: str
            One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
            default - `"DEBUG"`

    Returns
    -------
    logging.Logger
    """
    if not cfg:
        logging.config.dictConfig(LOGGING_DEFAULT_CONFIG)
    else:
        logging.config.dictConfig(cfg)

    logger = logger or logging.getLogger()

    if log_file or console:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(filename)s - %(funcName)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(getattr(logging, log_level))
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        if console:
            sh = logging.StreamHandler()
            sh.setLevel(getattr(logging, log_level))
            sh.setFormatter(formatter)
            logger.addHandler(sh)

    return logger


if __name__ == "__main__":
    pass
