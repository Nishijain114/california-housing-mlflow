import logging
import os
from datetime import datetime

def get_logger(name: str) -> logging.Logger:
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create a log filename with current date
    log_filename = datetime.now().strftime("logs/%Y-%m-%d.log")

    # Avoid adding multiple handlers if logger already has one
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter(
            "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger
