import os
import subprocess

# Define paths
TARGET_IMAGES_FOLDER = "/workspace/clipasso/target_images/"
NUM_STROKES = 8
GPU_NUM = 0

def process_images():
    """Runs CLIPasso on all images in the target_images directory without handling saving logic."""
    
    # Get list of images to process
    images = [f for f in os.listdir(TARGET_IMAGES_FOLDER) if f.endswith((".png", ".jpg", ".jpeg"))]

    if not images:
        print("❌ No images found in target_images/. Exiting...")
        return

    print(f"📂 Found {len(images)} images to process.")

    for filename in images:
        image_path = os.path.join(TARGET_IMAGES_FOLDER, filename)
        
        print(f"🎨 Processing {filename} with CLIPasso...")

        # Run CLIPasso without handling saving logic
        command = [
            "python3", "run_object_sketching.py",
            "--target_file", image_path,
            "--num_sketches", "1",
            "--mask_object", "0",
            "--fix_scale", "0",
            "--num_strokes", str(NUM_STROKES),
            "--multiprocess", "1",
            "--num_iter", "800",
            "--gpunum", str(GPU_NUM)
        ]

        subprocess.run(command, cwd="/workspace/clipasso")

    print("🎉 Processing complete!")

if __name__ == "__main__":
    process_images()
