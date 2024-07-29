"""
TAGS: color|colors|color palette|colour|colours|colour palette|dominant|extract|hex|image|palette|summarise|url|web|webpage|website
DESCRIPTION: Extract the most dominant colours from the given webpage 
REQUIREMENTS: pip install extcolors Pillow seleniumbase
REQUIREMENTS: This script requires function extract_colour_palette_from_image() from extract_colour_palette_from_image.py - you can get it here: https://github.com/J-sephB-lt-n/useful-code-snippets/blob/main/snippets/python/image/extract_colour_palette_from_image.py
REQUIREMENTS: Function extract_colour_palette_from_image() requires function arrange_images_in_grid() from arrange_images_in_grid.py - you can get it here: https://github.com/J-sephB-lt-n/useful-code-snippets/blob/main/snippets/python/image/pillow/arrange_images_in_grid.py
USAGE: python extract_colour_palette_from_webpage.py --url "https://en.wikipedia.org" --min_n_colours 3 --max_n_colours 10 --min_percent_of_image 0.01 --show_palette 
"""

import argparse
import time
from io import BytesIO
from collections import namedtuple

from PIL import Image
from seleniumbase import Driver

from extract_colour_palette_from_image import extract_colour_palette_from_image

Colour = namedtuple("Colour", "rgb hex percent_of_image")


def extract_colour_palette_from_webpage(
    url: str,
    min_n_colours: int,
    max_n_colours: int,
    min_percent_of_image: float,
    show_palette: bool,
):
    """Extracts the most prominent colours on the webpage

    Args:
        url (str): Webpage URL
        min_n_colours (int): Minimum number of colours allowed in output
        max_n_colours (int): Maximum number of colours allowed in output
        min_percent_of_image (float): Colours making up less than this percentage of total image are not included in the output
                                        This rule is overruled by `min_n_colours`
        show_palette (bool): If True, palette is displayed on the screen

    Returns:
        tuple[Colour, ...]: tuple of colours
    """
    driver = Driver(uc=True, headless=False)
    driver.uc_open_with_reconnect(url, 3)
    time.sleep(5)
    webpage_screenshot: bytes = driver.get_screenshot_as_png()
    driver.quit()
    webpage_image: Image.Image = Image.open(BytesIO(webpage_screenshot))

    return extract_colour_palette_from_image(
        pil_image=webpage_image,
        min_n_colours=min_n_colours,
        max_n_colours=max_n_colours,
        min_percent_of_image=min_percent_of_image,
        show_palette=show_palette,
    )


if __name__ == "__main__":
    # i.e. if script was run as command line tool
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-u",
        "--url",
        help="Webpage URL",
        type=str,
    )
    arg_parser.add_argument(
        "-n",
        "--min_n_colours",
        help="Minimum number of colours allowed in output",
        type=int,
    )
    arg_parser.add_argument(
        "-x",
        "--max_n_colours",
        help="Maximum number of colours allowed in output",
        type=int,
    )
    arg_parser.add_argument(
        "-p",
        "--min_percent_of_image",
        help="Colours making up less than this percentage of total image are not included in the output. This rule is overruled by --min_n_colours",
        type=float,
    )
    arg_parser.add_argument(
        "-s",
        "--show_palette",
        help="If this flag is included, palette is displayed on the screen",
        action="store_true",
    )
    args = arg_parser.parse_args()
    colour_palette = extract_colour_palette_from_webpage(
        url=args.url,
        min_n_colours=args.min_n_colours,
        max_n_colours=args.max_n_colours,
        min_percent_of_image=args.min_percent_of_image,
        show_palette=args.show_palette,
    )
    print("-- Extracted Colour Palette --")
    for colour in colour_palette:
        print(colour)
