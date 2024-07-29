"""
TAGS: beautifulsoup|content|extract|html|parse|text|user-facing|user|visible|web|webpage|website
DESCRIPTION: Extracts the text from an HTML string which a user would actually see in their browser (i.e. no javascript, CSS etc.)
USAGE: python extract_user_facing_text_from_html.py --input_filepath './input.html' --output_filepath './output.html' 
"""

import argparse
import re

from bs4 import BeautifulSoup


def extract_user_facing_text_from_html(html_str: str) -> str:
    """Extracts the text from an HTML string which a user would actually see in their browser
    (i.e. removes CSS, javascript etc.)"""
    soup = BeautifulSoup(html_str, "html.parser")

    # remove script elements #
    for element in soup(["style", "script"]):
        element.decompose()

    # remove non-visible elements #
    for element in soup.select('[style*="display:none"], [style*="visibility:hidden"]'):
        element.decompose()

    text = soup.get_text(separator="\n-----\n", strip=True)
    elements = text.split("\n-----\n")
    clean_elements = [re.sub(r"\s+", " ", x).strip() for x in elements]
    return "\n".join(clean_elements)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-i",
        "--input_filepath",
        help="path to HTML file to extract user-facing text from",
        required=True,
    )
    arg_parser.add_argument(
        "-o",
        "--output_filepath",
        help="Output will be written to this filepath",
        required=True,
    )
    args = arg_parser.parse_args()
    with open(args.input_filepath, "r", encoding="utf-8") as file:
        raw_html_str: str = file.read()
    user_facing_website_text: str = extract_user_facing_text_from_html(raw_html_str)
    with open(args.output_filepath, "w", encoding="utf-8") as file:
        file.write(user_facing_website_text)
