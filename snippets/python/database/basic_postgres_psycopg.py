"""
TAGS: client|database|db|postgres|postgresql|select|sql
DESCRIPTION: Basic usage examples of the python psycopg3 postgreSQL client
REQUIREMENTS: pip install psycopg
"""

import os
import psycopg

# Read from database #
my_query: str = (
    """
SELECT      my_col
FROM        my_table
;
            """.strip()
)


with psycopg.connect(
    conninfo=(
        "postgresql://"
        f'{os.environ["POSTGRES_USER"]}'
        f':{os.environ["POSTGRES_PASSWORD"]}'
        f"@localhost:5432"
    )
) as conn:
    with conn.cursor() as cur:
        cur.execute(my_query)
        query_result_as_list_of_tuples: list[tuple] = cur.fetchall()

    conn.row_factory = psycopg.rows.dict_row
    with conn.cursor() as cur:
        cur.execute(my_query)
        query_result_as_list_of_dicts: list[dict] = cur.fetchall()
