"""
TAGS: csv|data|save|write|writer
DESCRIPTION: Writing to a CSV file using the native python utility TODO
NOTES: I've included this as a user command in my neovim config: https://github.com/J-sephB-lt-n/dotfiles/blob/main/.config/nvim/lua/config/user_commands.lua
"""

import csv

with open("temp.csv", mode="w", encoding="utf-8") as file:
    csv_writer = csv.DictWriter(
        file,
        fieldnames=["name", "surname"],
        delimiter=",",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
    )
    csv_writer.writeheader()
    csv_writer.writerow({"name": "abraham", "surname": "lincoln"})
    csv_writer.writerow({"name": "oscar", "surname": "peterson"})
