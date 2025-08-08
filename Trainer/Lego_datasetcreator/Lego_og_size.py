import os
import shutil
import random

# Define paths
source_folder = "./Lego 256x256/photos"
generated_folder = "./Generated_Lego_datasets"
output_folder = "./Lego_og"

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Get all image names from the source folder
image_names = {f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))}

# Function to recursively search for matching images
def find_and_copy_images(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file in image_names:
                src_path = os.path.join(root, file)
                dest_path = os.path.join(output_folder, file)

                # If the file already exists, append a random number to the filename
                if os.path.exists(dest_path):
                    random_number = random.randint(1000, 9999)  # Generate a random number
                    name, ext = os.path.splitext(file)
                    new_file_name = f"{name}_{random_number}{ext}"
                    dest_path = os.path.join(output_folder, new_file_name)
                
                shutil.copy2(src_path, dest_path)
                print(f"Copied: {file} -> {dest_path}")

# Search and copy matching images
find_and_copy_images(generated_folder)

print("Processing complete. All matching images are copied to 'Lego_og'.")
