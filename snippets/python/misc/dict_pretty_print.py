"""
TAGS: dict|pretty|pretty print|print
DESCRIPTION: Function ppd() prints a dictionary in an aesthetic readable way using json.dumps()
"""

import json


def ppd(dct: dict) -> None:
    """Prints given dictionary in an aesthetic readable way using json.dumps()"""
    print(json.dumps(dct, indent=4, default=str))
