from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORS Middleware
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import re
import shutil
# apply  uvicorn main:app --reload   to run the backend

app = FastAPI()

# Allow React frontend to access FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (or specify ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths
PHOTO_DIR = Path("Lego_256x256/photos")
SKETCH_DIR = Path("Lego_256x256/sketchs")
PHOTO_DIR.mkdir(parents=True, exist_ok=True)
SKETCH_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/images")
async def get_images():
    """Returns a list of available images."""
    files = [f.name for f in PHOTO_DIR.glob("*.jpg")]
    return {"images": files}

@app.get("/image/{image_name}")
async def get_image(image_name: str):
    """Returns an image, doubled in size (512x512)."""
    image_path = PHOTO_DIR / image_name
    if not image_path.exists():
        return {"error": "Image not found"}
    
    # Open and convert to NumPy for OpenCV
    img = Image.open(image_path)
    img = np.array(img)  # Convert PIL to NumPy

    # Resize with OpenCV
    img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    # Convert back to PIL Image
    img_resized = Image.fromarray(img_resized)

    # Save and return resized image
    resized_path = f"static/resized_{image_name}"
    img_resized.save(resized_path)

    return FileResponse(resized_path)

@app.get("/sketchesnum")
async def get_sketchesnum():
    """Returns a list of available sketches and there size"""
    sketches = []
    for folder in SKETCH_DIR.iterdir():
        if folder.is_dir():
            num_files = sum(1 for f in folder.glob('**/*') if f.is_file())
            sketches.append({"folder": folder.name, "num_files": num_files})
    return {"sketches": sketches}
 


@app.post("/upload_sketch/{image_name}")
async def upload_sketch(image_name: str, file: UploadFile = File(...)):
    """Saves the sketched image in a folder with the same name as the sketch."""
    sketch_folder = SKETCH_DIR / re.sub(r"\.(jpeg|jpg|png)$", "",image_name, flags=re.IGNORECASE)
    sketch_folder.mkdir(parents=True, exist_ok=True) # Create folder if not exists
    sketch_path = sketch_folder / image_name # Save sketch in folder with same name as image
    
    with open(sketch_path, "wb") as f:
        f.write(await file.read())
    
    return {"message": "Sketch saved successfully", "filename": image_name}


@app.get("/")
async def root():
    return {"message": "FastAPI Backend is Running!"}

