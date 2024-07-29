"""
TAGS: css|extract|get|css|web|webpage|web page|website 
DESCRIPTION: Python function to extract all CSS from a webpage
REQUIREMENTS: pip install beautifulsoup4 requests
"""

import urllib.parse
from typing import Optional

import requests
from bs4 import BeautifulSoup


def extract_css_from_webpage(
    url: str, request_kwargs: Optional[dict] = None, verbose: bool = False
) -> tuple[list[str], list[str], list[dict]]:
    """Extracts CSS from webpage

    Args:
        url (str): Webpage URL
        request_kwargs (dict): These arguments are passed to requests.get() (when
                                fetching webpage HTML and external stylesheets)
        verbose (bool): Print diagnostic information

    Returns:
        tuple[ list[str], list[str], list[dict] ]: css_from_external_stylesheets, css_from_style_tags, inline_css
    """

    if not request_kwargs:
        request_kwargs = {
            "timeout": 10,
            "headers": {"User-Agent": "Definitely not an Automated Script"},
        }
    url_response = requests.get(url, **request_kwargs)
    if url_response.status_code != 200:
        raise requests.exceptions.HTTPError(
            f"received response [{url_response.status_code}] from [{url}]"
        )

    soup = BeautifulSoup(url_response.content, "html.parser")

    css_from_external_stylesheets: list[str] = []
    for link in soup.find_all("link", rel="stylesheet"):
        css_url = urllib.parse.urljoin(url, link["href"])
        if verbose:
            print(f"downloading external CSS stylesheet {css_url}")
        css_content: str = requests.get(css_url, **request_kwargs).text
        css_from_external_stylesheets.append(css_content)

    css_from_style_tags: list[str] = []
    for style_tag in soup.find_all("style"):
        css_from_style_tags.append(style_tag.string)

    inline_css: list[dict] = []
    for tag in soup.find_all(style=True):
        inline_css.append({"tag": str(tag), "css": tag["style"]})

    if verbose:
        print(
            f"""Extracted the following CSS from [{url}]:
    1. {len(css_from_external_stylesheets):,} external stylesheets (total {len("".join(css_from_external_stylesheets)):,} characters of text)
    2. {len(css_from_style_tags):,} style tags (total {len("".join(css_from_style_tags)):,} characters of text)  
    3. {len(inline_css):,} tags with inline CSS (total {len("".join( (x["css"] for x in inline_css) )):,} characters of text)  

"""
        )

    return css_from_external_stylesheets, css_from_style_tags, inline_css
