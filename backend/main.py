from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORS Middleware
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import re
import shutil
from datetime import datetime
from openai import OpenAI
import os
from io import BytesIO
from scipy.spatial.distance import cosine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
app = FastAPI()
# Securely fetch OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


# Helper functions
def convert_to_grayscale(image: Image.Image) -> np.ndarray:
    """Convert PIL image to grayscale numpy array."""
    image = image.convert("L")  # Convert to grayscale
    return np.array(image)

def compute_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """Computes SSIM similarity score between two images."""
    score, _ = ssim(image1, image2, full=True)
    return round(score * 10, 2)  # Scale SSIM from 0-10

def get_image_embedding(image_bytes: bytes):
    """Get image embeddings using OpenAI's text-embedding model (converts images to text first)."""
    response = client.embeddings.create(
        input=image_bytes.decode('latin1'),  # Convert bytes to a string
        model="text-embedding-3-small"  # ✅ Use a supported embedding model
    )
    return np.array(response.data[0].embedding)


#when needing to get the names of the files in the folder Before I changed the Labels to the names of the files
display_mapping = {}
save_mapping = {}
lego_txt_path = "lego.txt"
if Path(lego_txt_path).exists():
    with open(lego_txt_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if ":" in line:
                original, new_name = line.split(":", 1)
                display_mapping[new_name.strip()] = original.strip()
                save_mapping[original.strip()] = new_name.strip()
            elif line:
                display_mapping[line.strip()] = line.strip()
                save_mapping[line.strip()] = line.strip()

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
def list_images():
    """Returns a list of images with display names."""
    images = []
    for file in PHOTO_DIR.iterdir():
        if file.is_file():
            display_name = display_mapping.get(file.stem, file.stem)
            images.append({"filename": file.name, "display_name": display_name})
    return JSONResponse(content=images)

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
 

@app.post("/analyze_sketch/{image_name}")
async def analyze_sketch(image_name: str, sketch: UploadFile = File(...)):
    """Compares a sketch to its reference image using SSIM and OpenAI embeddings."""

    # Load images
    original_image = Image.open(PHOTO_DIR/ image_name)
    sketch_image = Image.open(BytesIO(await sketch.read()))

    # Convert to grayscale for SSIM comparison
    original_gray = convert_to_grayscale(original_image)
    sketch_gray = convert_to_grayscale(sketch_image)

    # Compute SSIM similarity score
    ssim_score = compute_ssim(original_gray, sketch_gray)

    # Convert images to bytes for OpenAI
    original_bytes = BytesIO()
    sketch_bytes = BytesIO()
    original_image.save(original_bytes, format="PNG")
    sketch_image.save(sketch_bytes, format="PNG")
    """
    # Compute OpenAI image embeddings
    original_embedding = get_image_embedding(original_bytes.getvalue())
    sketch_embedding = get_image_embedding(sketch_bytes.getvalue())

    # Compute cosine similarity
    cosine_similarity_score = round((1 - cosine(original_embedding, sketch_embedding)) * 10, 2)

    # Final Hybrid Score (weighted 50% each)
    final_score = round((ssim_score * 0.5) + (cosine_similarity_score * 0.5), 2)
    """
    return {
        "similarity_score_ssim": ssim_score,
        #"similarity_score_openai": cosine_similarity_score,
        "final_hybrid_score": ssim_score,
        "message": "Analysis complete"
    }

    
     

@app.post("/upload_sketch/{image_name}")
async def upload_sketch(image_name: str, file: UploadFile = File(...)):
    """Saves the sketched image in a folder with the same name as the sketch."""
    sketch_folder = SKETCH_DIR / re.sub(r"\.(jpeg|jpg|png)$", "",image_name, flags=re.IGNORECASE)
    sketch_folder.mkdir(parents=True, exist_ok=True) # Create folder if not exists
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # gets time date for unique file name
    file_extension = image_name.split(".")[-1]  # Get extension
    sketch_filename = f"{image_name}_{timestamp}.{file_extension}"
    sketch_path = sketch_folder / sketch_filename # Save sketch in folder with same name as image
    
    with open(sketch_path, "wb") as f:
        f.write(await file.read())
    
    return {"message": "Sketch saved successfully", "filename": image_name}


@app.get("/")
async def root():
    return {"message": "FastAPI Backend is Running!"}


