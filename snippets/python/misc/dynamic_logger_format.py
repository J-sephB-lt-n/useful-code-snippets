"""
TAGS: change|dynamic|format|log|logging|logger
DESCRIPTION: Code for dynamically changing the format of a python logger mid-script
NOTES: I have a more integrated example of this functionality here: https://github.com/J-sephB-lt-n/my-python-logging-setup
"""

import logging
from typing import Final

BASE_LOGGER_FORMAT: Final[str] = "%(asctime)s : %(name)s : %(levelname)s : %(message)s"

# set up a logger #
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(BASE_LOGGER_FORMAT))
logger.addHandler(handler)


def change_logger_format(logger: logging.Logger, new_format: str) -> None:
    """Changes the format of the provided logger (in every handler)"""
    new_formatter = logging.Formatter(new_format)
    for handler in logger.handlers:
        handler.setFormatter(new_formatter)


logger.info("Here is how log messages look at the beginning of the script")

for username in ("joe", "peter", "johann"):
    change_logger_format(
        logger=logger,
        new_format=BASE_LOGGER_FORMAT.replace(
            "%(message)s", f"[username={username}] %(message)s"
        ),
    )
    logger.info("loading user data")
    logger.info("validating user data")
    logger.info("transforming user data")
    logger.info("saving updated data to database")

change_logger_format(logger=logger, new_format=BASE_LOGGER_FORMAT)
logger.info("This is how log messages look at the end of the script")
