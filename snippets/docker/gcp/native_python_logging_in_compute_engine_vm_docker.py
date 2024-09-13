"""
TAGS: cloud|compute|compute engine|container|docker|engine|gcp|gcloud|google|google cloud|instance|log|logs|logger|logging|vm|virtual machine
DESCRIPTION: This code ensures that native python log messages get written to GCP Cloud Logging (and their format is respected)
REQUIREMENTS: pip install google-cloud-logging
"""

import logging
import sys

import google.cloud.logging

gcp_logging_client = google.cloud.logging.Client()
gcp_logging_handler = google.cloud.logging.handlers.CloudLoggingHandler(
    # writes to GCP Cloud Logging
    gcp_logging_client
)
console_logging_handler = logging.StreamHandler(sys.stdout)  # writes to standard out
file_logging_handler = logging.FileHandler("main.log")  # writes to local file
for handler in (gcp_logging_handler, console_logging_handler, file_logging_handler):
    handler.setFormatter(logging.Formatter("%(name)s - %(levelname)s - %(message)s"))
logger = logging.getLogger(__name__)
for handler in (gcp_logging_handler, console_logging_handler, file_logging_handler):
    logger.addHandler(handler)  # all logs will write to GCP Cloud Logging
logger.setLevel(logging.INFO)

# log all errors raised outside of a try/except except block
sys.excepthook = lambda err_type, err_value, err_traceback: logger.error(
    "Uncaught exception", exc_info=(err_type, err_value, err_traceback)
)

logger.info(
    "This message will write to GCP cloud logging, to main.log text file, and to standard out"
)

1 / 0  # the exception raised here, and the stack trace, will write to GCP cloud logging, to main.log text file, and to standard out
