import logging
import os



def init_logging(log_file=None, level=logging.INFO):
    """
    Set up logging configuration.
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler()]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=datefmt,
        handlers=handlers
    )
