"""
TAGS: auto|automation|bot|browser|browser automation|chrome|chromium|scrape|scraping|selenium|seleniumbase|web|web scraping
DESCRIPTION: Example of using an authenticated proxy server (e.g. for IP rotation) with selenium in python (chrome browser)
REQUIREMENTS: pip install seleniumbase # tested on seleniumbase==4.30.8
"""

import re
from types import SimpleNamespace

from seleniumbase import Driver

proxy_args = SimpleNamespace(
    username="your-proxy-server-username",
    password="your-proxy-server-password",
    port=6969,
    endpoint="your.proxy.server.domain",  # e.g. bright data's is "brd.superproxy.io",
)
proxy_args.auth_url = f"{proxy_args.username}:{proxy_args.password}@{proxy_args.endpoint}:{proxy_args.port}"

selenium_driver = Driver(uc=True, headless2=True, proxy=proxy_args.auth_url)

for _ in range(3):
    selenium_driver = Driver(uc=True, headless2=True, proxy=proxy_args.auth_url)
    selenium_driver.uc_open_with_reconnect("https://ipinfo.io/ip", 3)
    print(
        "your IP address is ",
        re.search(r"(\d+\.){3}\d+", selenium_driver.page_source).group(),
    )
    selenium_driver.close()
