"""
TAGS: domain|extract|get|link|re|regex|url|web|website
DESCRIPTION: Extracts domain part from given website URL
"""

import re


def extract_domain_from_website_url(url: str) -> str:
    """Extracts domain part of given website URL

    Example:
        >>> extract_domain_from_website_url("https://github.com/J-sephB-lt-n/mpd-tpd")
        'github.com'
    """
    return re.search(r"https?://([^/]+)", url).group(1)
