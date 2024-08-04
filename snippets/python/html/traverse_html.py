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
    include_terminal_tag_detail: bool,
    capture_attributes: tuple[str, ...] = ("class", "id"),
    tags_of_interest: Optional[tuple[str, ...]] = None,
) -> list[str]:
    """Recursively traverses (walks) through all tags in the provided HTML document,
    reporting the full path from the document root to that tag
    Args:
        soup_obj (bs4.BeautifulSoup): The HTML document, parsed into a BeautifulSoup object
        capture_attributes (tuple): These tag attributes will be reported in the output
                                    (for every tag in the path)
        tags_of_interest (tuple): If specified, only paths terminating with one of these
                                    tags will be included in the output
    """

    def recurse(tag, path) -> None:
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
        if include_terminal_tag_detail:
            full_repr: str = tag.prettify()
        else:
            full_repr = tag_repr
        if tags_of_interest is None or tag.name in tags_of_interest:
            paths.append(path + [full_repr])
        path.append(tag_repr)
        for child in tag.children:
            if child.name:  # if child element is a tag
                recurse(child, path)
        path.pop()

    paths = []
    recurse(soup_obj.html, [])
    return paths


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-u", "--url", help="URL of webpage to traverse", type=str, required=False
    )
    arg_parser.add_argument(
        "-o",
        "--output_path",
        help="Path to file where output will be written (JSON file)",
        type=str,
        required=False,
    )
    arg_parser.add_argument(
        "-t",
        "--include_terminal_tag_detail",
        help="If included, the full HTML of the terminal tag in each path is included",
        action="store_true",
    )
    arg_parser.add_argument(
        "-b",
        "--show_basic_example",
        help="Print a simple illustrative example to the terminal",
        action="store_true",
    )
    arg_parser.add_argument(
        "-c",
        "--show_complex_example",
        help="Print a comprehensive illustrative example to the terminal",
        action="store_true",
    )
    args = arg_parser.parse_args()
    if args.show_basic_example or args.show_complex_example:
        example_html: str = """
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
        print("Traversing the following HTML (example_html):\n", example_html)

    if args.show_basic_example:
        print(
            """>>> from bs4 import BeautifulSoup 
>>> soup = BeautifulSoup(example_html, "html5lib")
>>> dom_paths: list[str] = traverse_html(
...     soup, include_terminal_tag_detail=False
... )
>>> for path in dom_paths:
...     print(path)
        """
        )
        soup = BeautifulSoup(example_html, "html5lib")
        dom_paths: list[str] = traverse_html(soup, include_terminal_tag_detail=False)
        for path in dom_paths:
            print(path)
        print(
            """
>>> dom_paths_only_h1_tags: list[str] = traverse_html(
...     soup, include_terminal_tag_detail=False, tags_of_interest=("h1",)
... )
>>> for path in dom_paths_only_h1_tags:
...     print(path)
"""
        )
        dom_paths_only_h1_tags: list[str] = traverse_html(
            soup, include_terminal_tag_detail=False, tags_of_interest=("h1",)
        )
        for path in dom_paths_only_h1_tags:
            print(path)
        exit()
    if not args.url or not args.output_path:
        raise ValueError("Please provide both arguments `--url` and `--output_path`")

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
