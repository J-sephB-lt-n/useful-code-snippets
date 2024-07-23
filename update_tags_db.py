"""
Collects all tags from the top of all of the code scripts
in all folders and subfolders of /snippets/ and 
saves them in a searchable format in `tags_db.pickle`
"""

import os
import pickle
import re

tags_db: dict[str, dict] = {}

for dirpath, dirnames, filenames in os.walk("snippets"):
    for filename in filenames:
        filepath = f"{dirpath}/{filename}"
        print(f"processing {filepath}")
        with open(filepath, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file.readlines()):
                if line_num == 1:
                    tags_db[filepath] = {
                        # "tags": set(line.replace("TAGS: ", "").rstrip().split("|"))
                        "tags": set(
                            re.search(r"TAGS: (.*)", line)
                            .groups(1)[0]
                            .strip()
                            .split("|")
                        )
                    }
                elif line_num == 2:
                    # tags_db[filepath]["description"] = line.replace("DESCRIPTION: ", "").rstrip()
                    tags_db[filepath]["description"] = (
                        re.search("DESCRIPTION: (.*)", line).groups(1)[0].strip()
                    )

print("processed all files")
with open("tags_db.pickle", "wb") as file:
    pickle.dump(tags_db, file, pickle.HIGHEST_PROTOCOL)
