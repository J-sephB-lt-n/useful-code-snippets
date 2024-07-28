"""
TAGS: color|colors|color palette|colour|colours|colour palette|dominant|extract|hex|image|palette|summarise
DESCRIPTION: Extracts the most prominent colours in an image
REQUIREMENTS: pip install extcolors Pillow requests
REQUIREMENTS: The script arrange_images_in_grid.py is also required. You can get it here: https://github.com/J-sephB-lt-n/useful-code-snippets/blob/main/snippets/python/image/pillow/arrange_images_in_grid.py
"""

import argparse
from collections import namedtuple
from io import BytesIO

import extcolors
import requests
from PIL import Image

from arrange_images_in_grid import arrange_images_in_grid

Colour = namedtuple("Colour", "rgb hex percent_of_image")


def extract_colour_palette_from_image(
    pil_image: Image.Image,
    min_n_colours: int,
    max_n_colours: int,
    min_percent_of_image: float,
    show_palette: bool,
) -> tuple[Colour, ...]:
    """Extracts the most prominent colours in the provided image

    Args:
        pil_image (PIL.Image.Image): Input image
        min_n_colours (int): Minimum number of colours allowed in output
        max_n_colours (int): Maximum number of colours allowed in output
        min_percent_of_image (float): Colours making up less than this percentage of total image are not included in the output
                                        This rule is overruled by `min_n_colours`
        show_palette (bool): If True, palette is displayed on the screen

    Returns:
        tuple[Colour, ...]: tuple of colours
    """
    colours, total_pixel_count = extcolors.extract_from_image(pil_image)
    colours = sorted(colours, key=lambda x: x[1], reverse=True)
    colour_palette = []
    for colour in colours:
        if len(colour_palette) > max_n_colours:
            break
        rgb, pixel_count = colour
        if len(colour_palette) < min_n_colours or (
            pixel_count / total_pixel_count >= min_percent_of_image
        ):
            colour_palette.append(
                Colour(
                    rgb=rgb,
                    hex="#{:02x}{:02x}{:02x}".format(*rgb),
                    percent_of_image=pixel_count / total_pixel_count,
                )
            )
    if show_palette:
        arrange_images_in_grid(
            images=[
                Image.new(
                    mode="RGB",
                    size=(int(800 * palette_colour.percent_of_image), 100),
                    color=palette_colour.rgb,
                )
                for palette_colour in colour_palette
            ],
            layout=(len(colour_palette),),
            background_colour="white",
        ).show()

    return tuple(colour_palette)


if __name__ == "__main__":
    # script is executed as a command-line tool #
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-l",
        "--local_image_filepath",
        help="Path to image on local file system",
        type=str,
    )
    arg_parser.add_argument("-u", "--image_url", help="image web URL", type=str)
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
    if args.local_image_filepath and args.image_url:
        raise ValueError(
            "Please specify either `local_image_filepath` or `image_url`, not both"
        )
    if not args.local_image_filepath and not args.image_url:
        raise ValueError("Please provide either `local_image_filepath` or `image_url`")

    if args.local_image_filepath:
        image = Image.open(args.local_image_filepath)
    else:
        response = requests.get(
            args.image_url, timeout=10, headers={"User-Agent": "Let me in please"}
        )
        if response.status_code != 200:
            raise requests.exceptions.HTTPError(
                f"received response [{response.status_code}] from url [{args.image_url}]"
            )
        image = Image.open(BytesIO(response.content))

    colour_palette = extract_colour_palette_from_image(
        pil_image=image,
        min_n_colours=args.min_n_colours,
        max_n_colours=args.max_n_colours,
        min_percent_of_image=args.min_percent_of_image,
        show_palette=args.show_palette,
    )
    print("-- Extracted Colour Palette --")
    for colour in colour_palette:
        print(colour)
