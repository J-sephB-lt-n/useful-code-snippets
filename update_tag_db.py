"""
Collects all tags from the top of all of the code scripts
in all folders and subfolders of /snippets/ and 
saves them in a searchable format in `tags_db.pickle`
"""

import os

for (dirpath, dirnames, filenames) in os.walk("snippets"):
    for filename in filenames:
        print(dirpath, filename, sep="/")



