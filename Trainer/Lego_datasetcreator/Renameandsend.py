import os
import shutil

# Define source and destination folders
SOURCE_FOLDER = "C:/Users/zaccu/OneDrive/Documents/GitHub/U-SketchPlus/LegoDataset/Lego_256x256/photos"
DESTINATION_FOLDER = "C:/Users/zaccu/OneDrive/Documents/GitHub/U-SketchPlus/LegoDataset/Lego_256x256renamed/photos"

# Ensure destination exists
os.makedirs(DESTINATION_FOLDER, exist_ok=True)

def rename_and_copy():
    """Renames images by replacing spaces with underscores and copies them to the target folder."""
    images = [f for f in os.listdir(SOURCE_FOLDER) if f.endswith((".png", ".jpg", ".jpeg"))]

    if not images:
        print("❌ No images found in source folder.")
        return

    for filename in images:
        sanitized_filename = filename.replace(" ", "_")  # Replace spaces
        source_path = os.path.join(SOURCE_FOLDER, filename)
        destination_path = os.path.join(DESTINATION_FOLDER, sanitized_filename)

        shutil.copy(source_path, destination_path)
        print(f"✅ Copied: {filename} → {sanitized_filename}")

    print("🎉 All files renamed and copied successfully!")

if __name__ == "__main__":
    rename_and_copy()
