"""
TAGS: dict|pretty|pretty print|print
DESCRIPTION: Function ppd() prints a dictionary in an aesthetic readable way using json.dumps()
"""

import json
from pprint import pprint


def ppd(dct: dict) -> None:
    """Prints given dictionary in an aesthetic readable way using json.dumps()"""
    print(json.dumps(dct, indent=4, default=str))


if __name__ == "__main__":
    example_dict = {
        "users": {
            "23489hfdjkhsb4u31obfe": {
                "first_name": "abraham",
                "surname": "lincoln",
                "age": 215,
                "teams": ["dev", "uat", "review"],
                "access": {
                    "database": ["read", "write"],
                    "iam": ["read"],
                    "compute": ["read", "write", "delete"],
                },
            },
            "fghg8273gfeeidsgf329": {
                "first_name": "donald",
                "surname": "prumt",
                "age": 78,
                "access": {
                    "database": ["read"],
                    "iam": ["read", "write"],
                    "compute": ["delete"],
                },
            },
        }
    }
    print(example_dict)
    pprint(example_dict)
    ppd(example_dict)
