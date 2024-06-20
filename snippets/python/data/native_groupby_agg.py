"""
TAGS: agg|aggregation|column|columns|data|etl|elt|group|group by|groupby|itertools|library|native|row|rows|standard|standard lib|standard library|table|tabular|transform
DESCRIPTION: Example of grouping and aggregating tabular data using the python standard library 
"""

import itertools
import statistics
from typing import Generator

data = [
    # +------+------------+--------+
    # | id   | group      | amount |
    # +------+------------+--------+
    # | 11   |  control   | 27     |
    # | 12   |  treatment | 19     |
    # | 13   |  control   | 4      |
    # | 14   |  treatment | 15     |
    # | 15   |  treatment | 24     |
    # +------+------------+---------
    {"id": 11, "group": "control", "amount": 27},
    {"id": 12, "group": "treatment", "amount": 19},
    {"id": 13, "group": "control", "amount": 4},
    {"id": 14, "group": "treatment", "amount": 15},
    {"id": 15, "group": "treatment", "amount": 24},
]


def get_group(row: dict) -> str:
    """Returns the value in the 'group' column of the provided row of data"""
    return row["group"]


data.sort(key=get_group)

grouped_data = {
    group_name: list(group_rows)
    for group_name, group_rows in itertools.groupby(data, key=get_group)
}
# {
#   'control': [
#       {'amount': 27, 'group': 'control', 'id': 11},
#       {'amount': 4, 'group': 'control', 'id': 13}
#   ],
#   'treatment': [
#       {'amount': 19, 'group': 'treatment', 'id': 12},
#       {'amount': 15, 'group': 'treatment', 'id': 14},
#       {'amount': 24, 'group': 'treatment', 'id': 15}
#   ]
# }

# aggregations per group #
for agg_func in (sum, min, max, statistics.mean):
    print(agg_func.__name__)
    for group_name in grouped_data:
        amounts_in_group: Generator = (
            row["amount"] for row in grouped_data[group_name]
        )
        print("\t", group_name, agg_func(amounts_in_group))
# sum
#          control 31
#          treatment 58
# min
#          control 4
#          treatment 15
# max
#          control 27
#          treatment 24
# mean
#          control 15.5
#          treatment 19.333333333333332
