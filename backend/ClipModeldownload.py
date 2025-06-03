from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
"""
Modify file to Downalod different pretraiend models in specfic files.
Cache the model makes for faster loading times 
"""


MODEL_PATH = Path("models/clip_model")  # Define a local folder to store CLIP

# Ensure the folder exists
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Download & Save CLIP Model Locally
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=MODEL_PATH)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=MODEL_PATH)

# Save properly in `models/clip_model`
clip_model.save_pretrained(MODEL_PATH)
clip_processor.save_pretrained(MODEL_PATH)
