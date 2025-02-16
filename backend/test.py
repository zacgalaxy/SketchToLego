import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Function to Extract and Save Edges
def extract_and_save_edges(image_path, save_path="edges.png"):
    """
    Loads an image, applies Canny Edge Detection, and saves the edges.
    """
    # Load Image
    image = Image.open(image_path).convert("L")  # Convert to Grayscale
    image_np = np.array(image)

    # Apply Gaussian Blur to Reduce Noise
    blurred = cv2.GaussianBlur(image_np, (7,7), 3)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 150, 170)  # Adjust thresholds as needed

    # Save Edge Image
    cv2.imwrite(save_path, edges)

    # Display Original vs Edge Image
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image_np, cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(edges, cmap="gray")
    ax[1].set_title("Edge-detected Image")
    ax[1].axis("off")

    plt.show()
    print(f" Edge image saved as: {save_path}")

PHOTO_DIR = Path("Lego_256x256/photos")
# Example Usage
image_path = PHOTO_DIR/"Skull.jpg"  # Change this to your image file
output_path = "Skull.png"  # Where to save the edges
extract_and_save_edges(image_path, output_path)
