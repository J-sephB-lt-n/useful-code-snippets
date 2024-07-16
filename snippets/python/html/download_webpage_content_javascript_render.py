"""
TAGS: download|javascript|js|render|scrape|scraping|selenium|web|webpage|website 
DESCRIPTION: Downloads a webpage after opening it in a chrome browser (in order to render the javascript)
REQUIREMENTS: pip install seleniumbase
REQUIREMENTS: A chrome browser s required (see snippets/bash/ubuntu/install_chrome_browser_on_ubuntu.sh)
NOTES: On a machine without a display driver (e.g. gcloud VM), set USE_HEADLESS_BROWSER=True
"""

import time
from typing import Final

from seleniumbase import Driver

USE_HEADLESS_BROWSER: Final[bool] = True
RENDER_WAIT_NSECS: Final[int] = 2
URL: Final[str] = "https://amazon.com"
OUTPUT_PATH: Final[str] = "output_folder/output.html"

driver = Driver(uc=True, headless=USE_HEADLESS_BROWSER)
driver.uc_open_with_reconnect(URL, 3)
time.sleep(RENDER_WAIT_NSECS)
with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    file.write(driver.page_source)
driver.quit()
