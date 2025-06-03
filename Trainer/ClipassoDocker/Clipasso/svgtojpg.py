import os
import cairosvg

# Define source and destination directories
source_root = "8_stroke_sketch"
destination_root = "sketchs8strokes"

# Ensure the destination directory exists
os.makedirs(destination_root, exist_ok=True)

# Walk through the directory structure
for root, dirs, files in os.walk(source_root):
    if "best_iter.svg" in files:
        folder_name = os.path.basename(root)  # Extract the folder name
        svg_path = os.path.abspath(os.path.join(root, "best_iter.svg"))
        jpg_path = os.path.abspath(os.path.join(destination_root, f"{folder_name}.jpg"))  # Save directly in `sketchs20strokes/`

        if not os.path.exists(svg_path):
            print(f"⚠️ WARNING: SVG file not found: {svg_path}")
            continue

        print(f"✅ Converting {svg_path} → {jpg_path} with white background")

        try:
            # Convert SVG to JPG with a white background
            cairosvg.svg2png(url=svg_path, write_to=jpg_path, dpi=300, background_color="white")
            print(f"✅ Successfully converted: {svg_path} → {jpg_path}")
        except Exception as e:
            print(f"❌ Error converting {svg_path}: {e}")

print("✅ All files processed successfully.")
