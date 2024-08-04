"""
TAGS: dom|enumerate|html|path|paths|structure|tag|tags|traverse|tree|walk
DESCRIPTION: Walks recursively through entire HTML document, returning the full path leading to each tag.
REQUIREMENTS: pip install beautifulsoup4 html5lib requests 
USAGE: python enumerate_html_tag_paths.py --show_example 
USAGE: python enumerate_html_tag_paths.py --url TODO --output_path example_output.json --show_html_at_terminal_tag --tags_of_interest img,svg
"""

import argparse
from typing import Optional

from bs4 import BeautifulSoup
import requests


def enumerate_html_tag_paths(
    soup_obj: BeautifulSoup,
    show_html_at_terminal_tag: bool,
    capture_attributes: tuple[str, ...] = ("class", "id"),
    tags_of_interest: Optional[tuple[str, ...]] = None,
) -> list[list[str]]:
    """Recursively traverses (walks) through all tags in the provided HTML document,
    reporting the full path from the document root to that tag

    Args:
        soup_obj (bs4.BeautifulSoup): The HTML document, parsed into a BeautifulSoup object
        show_html_at_terminal_tag (bool): If true, shows the full HTML content of the last tag
                                            in each path
        capture_attributes (tuple): These tag attributes will be reported in the output
                                    (for every tag in the path)
        tags_of_interest (tuple): If specified, only paths terminating with one of these
                                    tags will be included in the output

    Returns:
        list[list[str]]: A list containing the path (from the root tag) to each tag in the HTML document
    """

    def recurse(tag, path) -> None:
        """This function processes `tag` and then calls itself recursively on each
        of this tag's children

        Args:
            tag: Current HTML tag being processed
            path: List that keeps track of the tag names from the root to the current tag

        Returns:
            None: recurse() appends contents to the `paths` list
        """
        tag_repr: str = f"<{tag.name}"
        for attribute in capture_attributes:
            if tag.get(attribute):
                if isinstance(tag.get(attribute), list):
                    tag_repr += f' {attribute}="{" ".join(tag.get(attribute))}"'
                else:
                    tag_repr += f' {attribute}="{tag.get(attribute)}"'
        tag_repr += ">"
        if show_html_at_terminal_tag:
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
        "-s",
        "--show_html_at_terminal_tag",
        help="Show the full HTML content of the last tag in each path",
        action="store_true",
    )
    arg_parser.add_argument(
        "-t",
        "--tags_of_interest",
        help="If specified, only paths terminating with one of these tags will be included in the output",
    )
    arg_parser.add_argument(
        "-e",
        "--show_example",
        help="Print a simple illustrative example to the terminal",
        action="store_true",
    )
    args = arg_parser.parse_args()
    if args.show_example:
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
        print(
            """>>> from bs4 import BeautifulSoup 
>>> soup = BeautifulSoup(example_html, "html5lib")
>>> dom_paths: list[list[str]] = enumerate_html_tag_paths(
...     soup, show_html_at_terminal_tag=False
... )
>>> for path in dom_paths:
...     print(path)
        """
        )
        soup = BeautifulSoup(example_html, "html5lib")
        dom_paths: list[list[str]] = enumerate_html_tag_paths(
            soup, show_html_at_terminal_tag=False
        )
        for path in dom_paths:
            print(path)
        print(
            """
>>> dom_paths_only_h1_tags: list[list[str]] = enumerate_html_tag_paths(
...     soup, show_html_at_terminal_tag=True, tags_of_interest=("h1",)
... )
>>> for path in dom_paths_only_h1_tags:
...     for depth, tag in enumerate(path):
...         [print("\\t" * depth + line) for line in tag.split("\\n")]
...     print("---")
"""
        )
        dom_paths_only_h1_tags: list[list[str]] = enumerate_html_tag_paths(
            soup, show_html_at_terminal_tag=True, tags_of_interest=("h1",)
        )
        for path in dom_paths_only_h1_tags:
            for depth, tag in enumerate(path):
                [print("\t" * depth + line) for line in tag.split("\n")]
            print("---")
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
    tag_paths: list[list[str]] = enumerate_html_tag_paths(
        soup_obj=soup,
        show_html_at_terminal_tag=args.show_html_at_terminal_tag,
        tags_of_interest=args.tags_of_interest,
    )
