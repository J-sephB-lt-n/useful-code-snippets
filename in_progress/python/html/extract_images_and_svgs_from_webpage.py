"""
TAGS: extract|graphic|graphics|image|images|svg|vector|web|webpage|website
DESCRIPTION: Extracts all kinds of different images from a given webpage (see function docstring for details)
REQUIREMENTS: pip install beautifulsoup4 cairosvg html5lib Pillow requests 
"""

import argparse

import requests
from bs4 import BeautifulSoup
from PIL import Image


def traverse_html(soup: BeautifulSoup):
    """TODO"""


def extract_images_svgs_from_html(html: bytes) -> tuple[Image.Image, ...]:
    """docstring TODO

    Notes:
        These specific cases are implemented:
            1. <img> tag containing a 'src' attribute
            2. <img> tag containing image encoded as base64 (i.e. in a "data:" url)
            3. <svg> tag to draw image in the browser ("inline svg")
            4. <embed> tag containing 'src' attribute
            5. <picture> tag with a srcset attribute
    """
    soup = BeautifulSoup(html, "html5lib")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-u", "--url", "Webpage URL", type=str)
    args = arg_parser.parse_args()

    webpage_response = requests.get(
        url=args.url, timeout=10, headers={"User-Agent": "Definitely not a Script"}
    )
    extracted_images = extract_images_svgs_from_html(webpage_response.content)
