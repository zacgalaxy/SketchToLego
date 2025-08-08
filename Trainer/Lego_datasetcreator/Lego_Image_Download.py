import os
import requests
import pandas as pd

# Load the spreadsheet (replace 'your_spreadsheet.csv' with your file name)
spreadsheet_file = "lego_sets_and_themes.csv"  # Replace with your file name
output_directory = os.path.join(os.path.dirname(os.path.abspath(spreadsheet_file)), "images 4 (Flowers)")

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Load the spreadsheet (supports .csv only for this case)
df = pd.read_csv(spreadsheet_file)

# Ensure the spreadsheet has the required columns
required_columns = ['set_number', 'set_name', 'year_released', 'number_of_parts', 'image_url', 'theme_name']
if not all(column in df.columns for column in required_columns):
    raise ValueError(f"The spreadsheet must contain the following columns: {', '.join(required_columns)}")
intrested_theme= ['Archtiecture', 'Skylines' , 'pick A model' , 'Creator 3-in-1']
nature=["flowers","botanical"]

# Open the file and read its contents into a list
objects_to_test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'objectstotest.txt')
with open(objects_to_test_file, 'r') as file:
    objectstotest = []
    for line in file.readlines():
        # Split lines containing underscores into separate words and normalize
        words = line.strip().replace("_", " ").split()
        objectstotest.extend(word.lower() for word in words)



# Normalize strings for comparison
def normalize_string(s):
    string=str(s)
    return string.strip().lower()

#Preprocess the interested themes
normalized_interested_theme = [normalize_string(theme) for theme in intrested_theme]

def randomimagegeneration(file_name, url, year, theme, output_dir ):
    try:
        # Create theme-specific directory
        year_dir = os.path.join(output_dir, year)
        os.makedirs(year_dir, exist_ok=True)

        # Create the full path to save the file
        file_path = os.path.join(year_dir, f"{file_name}.jpg")

        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        # Write the image content to the file
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

        print(f"Downloaded: {file_name}.jpg to {year_dir}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

    print("All images processed.")




# Function to download a single image
def download_image(file_name, url,theme, output_dir ):
    try:
        # Create theme-specific directory
        theme_dir = os.path.join(output_dir, theme)
        os.makedirs(theme_dir, exist_ok=True)

        # Create the full path to save the file
        file_path = os.path.join(theme_dir, f"{file_name}.jpg")

        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

        # Write the image content to the file
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

        print(f"Downloaded: {file_name}.jpg to {theme_dir}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

# Download all images from the URLs in the spreadsheet
for index, row in df.iterrows():
    year= row['year_released']
    url = row['image_url']
    file_name = row['set_name']
    theme= row['theme_name']
    
    
    #Normalize the theme name for comparison
    normalized_theme = normalize_string(theme)
    
    #normalized set name
    setnname= normalize_string(file_name)
    
    #if pd.notna(url) and pd.notna(file_name) and year > 2020 and any(obj in setnname for obj in objectstotest):
    #if pd.notna(url) and pd.notna(file_name) and normalized_theme in normalized_interested_theme:  # Filter by year and theme
    if pd.notna(url) and pd.notna(file_name) and any(obj in setnname for obj in nature):
        sanitized_file_name = file_name.replace("/", "-").replace("\\", "-")  # Avoid invalid characters
        download_image(sanitized_file_name, url ,theme, output_directory)

random_sample = df.sample(n=100, random_state=42)  # Use `random_state` for reproducibility
for _, row in random_sample.iterrows():
    year= row['year_released']
    url = row['image_url']
    file_name = row['set_name']
    theme= row['theme_name']
    randomimagegeneration(file_name, url, year, theme, output_directory)      

print("All images processed.")
