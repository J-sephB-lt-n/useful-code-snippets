"""
TAGS: browser|bucket|cloud|cloud storage|dev|gcloud|gcp|google|google cloud|headless|scrape|scraping|selenium|storage|web
DESCRIPTION: Utility class for debugging headless selenium in a google cloud environment
REQUIREMENTS: pip install google-cloud-storage seleniumbase
"""

import datetime
from typing import Optional

import google.cloud.storage
from seleniumbase import Driver


class SeleniumOnGcpDevUtil:
    """
    Utility for debugging headless selenium in a google cloud environment
    (e.g. on a VM or Cloud Run, where there is no display driver present)

    Example:
        >>> selenium_dev = SeleniumOnGcpDevUtil(gcp_output_bucket_name="your-bucket-name", selenium_driver_kwargs={"uc": True, "headless": True})
        >>> selenium_dev.driver.uc_open_with_reconnect("https://www.google.com", 3)
        >>> selenium_dev.save_screenshot_to_bucket("html") # by default saves to <current local datetime>.html
        >>> selenium_dev.save_screenshot_to_bucket("png") # by default saves to <current local datetime>.png
    """

    def __init__(
        self,
        gcp_output_bucket_name: str,
        selenium_driver_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Args:
            gcp_output_bucket_name (str): Name of cloud storage bucket where screenshots are written to
            selenium_driver_kwargs (dict, optional): Keyword arguments to pass to selenium Driver instance
        """
        if selenium_driver_kwargs is None:
            selenium_driver_kwargs = {"uc": True, "headless": True}
        self.driver = Driver(**selenium_driver_kwargs)
        self.gcp_storage_client = google.cloud.storage.Client()
        self.gcp_output_bucket = self.gcp_storage_client.bucket(gcp_output_bucket_name)

    def save_screenshot_to_bucket(
        self,
        screenshot_type="html",
        output_filepath: Optional[str] = None,
    ) -> None:
        """
        Saves view of current browser state to a file in a google cloud storage bucket

        Args:
            screenshot_type (str, optional): Content type of browser screenshot - must be one of {'html', 'png'}. Default is 'html'
            output_filepath (str, optional): Output will be written to the filepath on the bucket e.g. output_filepath="my/desired/filepath/file.html"
                                                Defaults to '<current local datetime>.<screenshot_type>'
        """
        if output_filepath is None:
            output_filepath = datetime.datetime.now().strftime(
                f"%Y_%m_%d__%H_%M_%S.{screenshot_type}"  # uses local timezone
            )
        match screenshot_type:
            case "html":
                self.gcp_output_bucket.blob(output_filepath).upload_from_string(
                    data=self.driver.page_source,
                    content_type="text/html",
                )
            case "png":
                self.gcp_output_bucket.blob(output_filepath).upload_from_string(
                    data=self.driver.get_screenshot_as_png(),
                    content_type="image/png",
                )
            case _:
                raise ValueError("screenshot_type must be one of {'html', 'png'}")
