"""
TAGS: dict|pretty|pretty print|print
DESCRIPTION: Function pd() prints a dictionary in an aesthetic readable way using json.dumps()
"""

import json


def pd(dct: dict) -> None:
    """Prints given dictionary in an aesthetic readable way using json.dumps()"""
    print(json.dumps(dct, indent=4, default=str))
