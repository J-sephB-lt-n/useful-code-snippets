#
"""
Free-text search for a code snippet
"""

import json
import pickle
import sys
from collections import defaultdict

if len(sys.argv) == 2:
    search_text = sys.argv[1]
else:
    search_text: str = input(
        """
    +-------------------------+
    | SEARCH FOR CODE SNIPPET |
    +-------------------------+
Please describe what you are looking for:
    - Your search query is cleaned and compared with snippet tags for 
        word overlap.
    - You can preserve phrases using underscores e.g. 
            the query 'cloud storage google_cloud'
            results in the lookup tokens {"cloud", "storage", "google cloud"}
        """
    )
search_tokens: set[str] = [
    word.lower().strip().replace("_", " ") for word in search_text.split()
]

with open("tags_db.pickle", "rb") as file:
    tags_db = pickle.load(file)

print("Search tokens: ", search_tokens)
matches = {}
for filepath, info in tags_db.items():
    intersection: set[str] = info["tags"].intersection(search_tokens)
    if len(intersection) > 0:
        matches[filepath] = intersection

if len(matches) == 0:
    print(" -- no code snippets found matching input query --")
else:
    matches = dict(sorted(matches.items(), key=lambda x: (-len(x[1]), x[0])))
    print(
        """
+----------------+
| SEARCH RESULTS |
+----------------+
(sorted by number of matching keywords)"""
    )
    print(json.dumps(matches, indent=4, default=str))
