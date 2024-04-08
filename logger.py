import logging
from logging.handlers import RotatingFileHandler

def setup_logger():
    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create console handler and set level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create rotating file handler and set level to INFO
    file_handler = RotatingFileHandler('app.log', maxBytes=1024000, backupCount=3)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Logger setup completed.")
