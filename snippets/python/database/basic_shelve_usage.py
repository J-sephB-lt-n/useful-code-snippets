"""
TAGS: data|database|dbm|dict|dictionary|key|key value|keyvalue|local|redis|shelve|sqlite|value
DESCRIPTION: Shows basic usage of the native python `shelve` package
NOTES: `shelve` stores values containig complex data types (dicts, classes etc.) using `pickle`
NOTES: In a multi-threaded environment, concurrent reads are safe but not concurrent writes (see thread-safe code below)
NOTES: The native python package `dbm` provides a similar but performant, restricted and lower-level interface
"""

import shelve
from typing import Any, Final

DB_PATH: Final[str] = "temp"

# create a file in the current folder called 'temp.db' and store some data in it
with shelve.open(
    DB_PATH,
    flag="c",  # open database for reading and writing, creating it if it doesn’t exist.
    writeback=True,  # writeback=False disallows mutating values in place (e.g. using list .append()), but writeback=True only persists data on db.sync() or db.close()
) as db:  #
    # Available flags (open modes) are:
    #   'r' (default): Open existing database for reading only.
    #   'w': Open existing database for reading and writing.
    #   'c': Open database for reading and writing, creating it if it doesn’t exist.
    #   'n': Always create a new, empty database, open for reading and writing (overwrites existing)
    db["users"] = {}
    db["users"]["joe"] = {"age": 37, "rank": "servant"}
    db["users"]["jesse"] = {"age": 3, "rank": "king"}

with shelve.open(DB_PATH, "r") as db:
    all_users: dict = db["users"]

print(all_users)
# {'joe': {'age': 37, 'rank': 'servant'}, 'jesse': {'age': 3, 'rank': 'king'}}

#################################################
# Thread-safety in a multi-threaded environment #
#################################################
import threading


def thread_safe_shelve_write(lock: threading.Lock, db_path: str, key: str, value: Any):
    with lock:
        with shelve.open(db_path, "w", writeback=False) as db:
            db[key] = value


lock = threading.Lock()
threading.Thread(
    target=thread_safe_shelve_write,
    kwargs={"lock": lock, "db_path": DB_PATH, "key": "status", "value": "success"},
).start()

with shelve.open(DB_PATH, "r") as db:
    for key, value in db.items():
        print(f"{key} : {value}")
# users : {'joe': {'age': 37, 'rank': 'servant'}, 'jesse': {'age': 3, 'rank': 'king'}}
# status : success
