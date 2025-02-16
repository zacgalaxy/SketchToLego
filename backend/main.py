from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORS Middleware
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import re
from transformers import CLIPProcessor, CLIPModel
from skimage.metrics import structural_similarity as ssim
import torch
from datetime import datetime
from openai import OpenAI
import os
from io import BytesIO
from scipy.spatial.distance import cosine
from dotenv import load_dotenv


app = FastAPI()

# Define paths
PHOTO_DIR = Path("Lego_256x256/photos")
PHOTO_DIR2= Path("Lego_512x512/photos")
SKETCH_DIR = Path("Lego_256x256/sketchs")
PHOTO_DIR.mkdir(parents=True, exist_ok=True)
SKETCH_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path("models/clip_model")
inappropriate_labels = [
    "a violent image", 
    "a dick",
    "a flaccid object" 
    "explicit content", 
    "offensive symbols",
    "gore",
    "hateful imagery"
]
@app.on_event("startup")
async def load_clip_model():
    global clip_model, clip_processor
    try:
        clip_model = CLIPModel.from_pretrained(MODEL_PATH)
        clip_processor = CLIPProcessor.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading CLIP model: {e}")

#Greyscale and SSIM did not work 

# Convert image to edges using Canny Edge Detection
def extract_edgesimage(image: Image.Image) -> np.ndarray:
    """Convert an image to edges using Canny Edge Detection."""
    image = image.convert("L")  # Convert to grayscale
    image_np = np.array(image)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image_np, (7, 7), 3)
    
    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 150, 170)  # Adjust thresholds as needed
    
    return edges

def extract_edgessketch(image: Image.Image) -> np.ndarray:
    """Convert an image to edges using Canny Edge Detection."""
    image = image.convert("L")  # Convert to grayscale
    image_np = np.array(image)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image_np, (5, 5), 0)
    
    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 100)  # Adjust thresholds as needed
    
    return edges

def compute_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
    """Computes SSIM similarity score between two images."""
    score, _ = ssim(image1, image2, full=True)
    return round(score * 10, 2)  # Scale SSIM from 0-10

def get_image_embedding(image: Image.Image):
    """Compute CLIP embeddings for an image."""
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embedding = clip_model.get_image_features(inputs["pixel_values"])
    return embedding.squeeze().numpy()

def get_text_embeddings(text_list):
    """Computes CLIP embeddings for multiple text labels."""
    text_list = [f"rough sketch of a {text}" for text in text_list]  # Ensure it's a list of strings
    inputs = clip_processor(text=text_list, return_tensors="pt", padding=True,truncation=True )
    with torch.no_grad():
        embeddings = clip_model.get_text_features(**inputs)
    return embeddings.squeeze().numpy()

def get_text_embedding(text):
    """Computes CLIP embeddings for a single text label."""
    text=f"rough sketch of a {text} on white background"
    print(text)
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embedding = clip_model.get_text_features(**inputs)
    return embedding.squeeze().numpy()  # Returns a single 1D NumPy array


def detect_inappropriate_content(image: Image.Image):
    """Checks if an image is similar to any inappropriate category."""
    image_embedding = get_image_embedding(image)
    text_embeddings = get_text_embeddings(inappropriate_labels)
    # Compute cosine similarity with each label
    scores = [1 - cosine(image_embedding, text_emb) for text_emb in text_embeddings]

    # Get max similarity score
    max_score = max(scores)
    best_match = inappropriate_labels[scores.index(max_score)]

    return best_match, max_score



def clipvsssim():
    return None

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
    image_path = PHOTO_DIR2 / image_name
    if not image_path.exists():
        return {"error": "Image not found"}

    return FileResponse(image_path)

@app.get("/sketchesnum")
async def get_sketchesnum():
    """Returns a list of available sketches and there size"""
    sketches = []
    for folder in SKETCH_DIR.iterdir():
        if folder.is_dir():
            num_files = sum(1 for f in folder.glob('**/*') if f.is_file())
            sketches.append({"folder": folder.name, "num_files": num_files})
    return {"sketches": sketches}
 
"""
#Fun try to use clips textual emeddings and image emeddings work togther to 
@app.post("/analyze_sketch/{image_name}")
async def analyze_sketch(image_name: str, sketch: UploadFile = File(...)):

    # Read file once and store it
    file_bytes = await sketch.read()
    sketch_image = Image.open(BytesIO(file_bytes))
    best_match, score = detect_inappropriate_content(sketch_image)
    threshold = 0.7
    print(score)
    print(best_match)
    if score > threshold:
        return {"flagged": True, "reason": best_match, "score": score}
  
    # Define a textual description (Can be made dynamic)
    text_description = f"{image_name.split('.')[0]}"  
    text_embedding = get_text_embedding(text_description)
    sketch_embedding = get_image_embedding(sketch_image)

    # Compute cosine similarity
    cosine_similarity_score = float(round((1 - cosine(text_embedding, sketch_embedding)) * 10, 2))
    print(f" cosign sim  {cosine_similarity_score}")
    return {
        "similarity_score_clip": cosine_similarity_score,
        "message": "Analysis complete (Text-to-Sketch)"
    }

"""


@app.post("/analyze_sketch/{image_name}")
async def analyze_sketch(image_name: str, sketch: UploadFile = File(...)):
    """Compares a sketch to its reference image using SSIM and OpenAI embeddings."""

    # Load images
    original_image = Image.open(PHOTO_DIR/ image_name)
    sketch_image = Image.open(BytesIO(await sketch.read()))
    
    """Usage for experementation later will not decided anything"""
    # Convert to edges for SSIM comparison
    original_edges = extract_edgesimage(original_image)
    sketch_edges = extract_edgessketch(sketch_image)

    # Compute SSIM similarity score on edges
    ssim_score = float(compute_ssim(original_edges, sketch_edges))
    
    # Compute CLIP embeddings
    original_embedding = get_image_embedding(original_image)
    sketch_embedding = get_image_embedding(sketch_image)


    # Compute cosine similarity
    cosine_similarity_score = float(round((1 - cosine(original_embedding, sketch_embedding)) * 10, 2))
    print(f" ssim {ssim_score}")
    print(f" cosign sim  {cosine_similarity_score}")
    return {
        "similarity_score_edge_ssim": ssim_score,
        "similarity_score_clip": cosine_similarity_score,
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


