import cv2
import numpy as np
import os
from screeninfo import get_monitors  # Get screen resolution dynamically

# Get system's screen resolution and use half of it
monitor = get_monitors()[0]  # Get primary monitor
screen_width = monitor.width // 2
screen_height = monitor.height // 2

# Folder paths
input_folder = "./Lego_256x256renamed/photos"
output_folder = "./renamededitedLego_256x256"
os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists

# Global variables
drawing = False  # True when the eraser is being used
eraser_size = 20  # Size of the eraser
image = None  # Original full-resolution image
resized_image = None  # Image resized to fit the screen
clone = None  # Backup of the resized image
scale_x = 1
scale_y = 1

def resize_image(image, max_width, max_height):
    """Resize image while maintaining aspect ratio."""
    global scale_x, scale_y
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)  # Scale factor to fit within the screen
    new_w, new_h = int(w * scale), int(h * scale)
    scale_x, scale_y = w / new_w, h / new_h  # Store scale to map eraser strokes
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def erase(event, x, y, flags, param):
    """ Eraser function to remove parts of the image """
    global drawing, resized_image, image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True  # Start erasing

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Map eraser coordinates to original image
        orig_x, orig_y = int(x * scale_x), int(y * scale_y)
        cv2.circle(image, (orig_x, orig_y), int(eraser_size * scale_x), (255, 255, 255), -1)
        # Also erase on the resized image for display
        cv2.circle(resized_image, (x, y), eraser_size, (255, 255, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False  # Stop erasing

# Get list of all images in the input folder
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
image_files.sort()  # Sort images for consistency

for image_name in image_files:
    image_path = os.path.join(input_folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image {image_name}. Skipping...")
        continue

    # Resize the image to fit the screen and store original scale
    resized_image = resize_image(image.copy(), screen_width, screen_height)
    clone = resized_image.copy()  # Store a copy for reset

    cv2.namedWindow("Sketch Canvas", cv2.WINDOW_NORMAL)  # Allow resizing
    cv2.setMouseCallback("Sketch Canvas", erase)

    print(f"Editing {image_name}...")

    while True:
        cv2.imshow("Sketch Canvas", resized_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):  # Press 'r' to reset the image
            print("Resetting image...")
            image = cv2.imread(image_path)  # Reload original image
            resized_image = resize_image(image.copy(), screen_width, screen_height)
        elif key == ord('s'):  # Press 's' to save and move to next image
            save_path = os.path.join(output_folder, image_name)
            cv2.imwrite(save_path, image)  # Save original resolution
            print(f"Saved {image_name} to {output_folder}")
            break
        elif key == 27:  # Press 'ESC' to exit the program
            print("Exiting...")
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()  # Close window before moving to the next image

print("All images processed.")
