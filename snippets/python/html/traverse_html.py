"""
TAGS: html|path|tags|traverse|walk
DESCRIPTION: Walk recursively through an entire HTML document. This is typically useful when you want to process certain tags, and want to retain that tags position information within the overall document hierarchy (DOM). 
REQUIREMENTS: pip install beautifulsoup4 html5lib requests 
USAGE: python traverse_html.py --url https://en.wikipedia.org/wiki/Art --output_path './output_example.json'
"""

import argparse
from typing import Optional

from bs4 import BeautifulSoup
import requests


def traverse_html(
    soup_obj: BeautifulSoup,
    capture_attributes: tuple[str, ...] = ("class", "id"),
) -> list[str]:
    """Recursively traverses (walks) through all tags in the provided HTML document,
    reporting the full path from the document root to that tag
    Args:
        soup_obj (bs4.BeautifulSoup): The HTML document, parsed into a BeautifulSoup object
        capture_attributes (tuple): These tag attributes will be reported in the output
                                    (for every tag in the path)
    """

    def recurse(tag, path):
        """This function processes `tag` and then calls itself recursively on each
        of this tag's children
        Args:
            tag: Current HTML tag being processed
            path: List that keeps track of the tag names from the root to the current tag
        """
        tag_repr: str = f"<{tag.name}"
        for attribute in capture_attributes:
            if tag.get(attribute):
                if isinstance(tag.get(attribute), list):
                    tag_repr += f' {attribute}="{" ".join(tag.get(attribute))}"'
                else:
                    tag_repr += f' {attribute}="{tag.get(attribute)}"'
        tag_repr += ">"
        path.append(tag_repr)
        paths.append(path[:])
        for child in tag.children:
            if child.name:  # if child element is a tag
                recurse(child, path)
        path.pop()

    paths = []
    recurse(soup_obj.html, [])
    return paths


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-u", "--url", help="URL of webpage to traverse", type=str)
    arg_parser.add_argument(
        "-o",
        "--output_path",
        help="Path to file where output will be written (JSON file)",
        type=str,
    )
    args = arg_parser.parse_args()
    url_response = requests.get(
        args.url,
        timeout=10,
        headers={"User-Agent": "Definitely not an Automated Script"},
    )
    if url_response.status_code != 200:
        raise requests.exceptions.HTTPError(
            f"Received response [{url_response.status_code}] from {args.url}"
        )
    soup = BeautifulSoup(url_response.content, "html5lib")
    dom: list = traverse_html(soup)

"""
<html>
    <head>
        <title>Page Title</title>
    </head>
    <body>
        <nav>
            <h1 class="black-text" id="navbar-title">Navbar Title Text</h1>
        </nav>
        <div id="main-content">
            <h1 class="main-content-header">Hello</h1>
            <p class="main-content-body">For as long as I can remember</p>
        </div>
    </body>
</html>
"""
