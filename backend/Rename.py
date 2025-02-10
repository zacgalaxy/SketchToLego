import os
from pathlib import Path

# Define paths (Modify if needed)
photos_dir = Path("Lego_256x256/photos")  # Path to the folder with images
lego_txt_path = "lego.txt"  # Path to your label text file

# Ensure the directory exists
if not photos_dir.exists():
    print(f"Error: Directory '{photos_dir}' not found.")
    exit(1)

# Read the label mappings from lego.txt
mapping = {}
with open(lego_txt_path, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if ":" in line:
            original, new_name = line.split(":", 1)
            mapping[original.strip()] = new_name.strip()
        elif line:  # Items without a rename mapping, use the same name
            mapping[line.strip()] = line.strip()

# Rename files based on mapping
for file in photos_dir.iterdir():
    if file.is_file():  # Ensure it's a file
        for key, new_label in mapping.items():
            if key in file.stem:  # Check if filename contains the keyword
                new_filename = f"{new_label}{file.suffix}"  # Keep same extension
                new_filepath = photos_dir / new_filename
                os.rename(file, new_filepath)
                print(f"Renamed: {file.name} -> {new_filename}")
                break  # Stop checking after the first match

print("Renaming process completed.")