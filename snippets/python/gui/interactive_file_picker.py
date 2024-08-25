"""
TAGS: gui|file|file picker|picker|tkinter
DESCRIPTION: Opens an interactive file-picker GUI using tkinter (i.e. native python)
"""

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

file_path: str = filedialog.askopenfilename()
print(file_path)
