import logging
from logging.handlers import RotatingFileHandler

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    file_handler = RotatingFileHandler(filename=name + ".log", maxBytes=1024 * 1024, backupCount=5)
    # Create a log format using Log Record attributes
    fmt = logging.Formatter("%(filename)s: %(asctime)s | %(levelname)s | %(lineno)s >>>> %(message)s")
    # Add each handler to the Logger object
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
