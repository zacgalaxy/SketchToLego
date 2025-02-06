import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { ReactSketchCanvas } from "react-sketch-canvas";

const API_URL = "http://127.0.0.1:8000"; // FastAPI backend URL

const App = () => {
  const [images, setImages] = useState([]); // List of images
  const [selectedImage, setSelectedImage] = useState(null); // Current image
  const [imageSrc, setImageSrc] = useState(""); // Image URL for viewing
  const sketchRef = useRef(null); // Reference to sketchpad

  // Fetch list of images from FastAPI
  useEffect(() => {
    axios.get(`${API_URL}/images`).then((response) => {
      setImages(response.data.images);
    });
  }, []);

  // Load selected image
  const loadImage = (imageName) => {
    setSelectedImage(imageName);
    setImageSrc(`${API_URL}/image/${imageName}`);
  };
  // add white background
  const whitebackgrond
  // Save the sketch
  const saveSketch = async () => {
    if (!sketchRef.current || !selectedImage) return;

    // Convert sketch to PNG format
    const sketchData = await sketchRef.current.exportImage("");
    
    const imageName = selectedImage; // Save with same name as the image

    // Convert base64 data to File object
    const formData = new FormData();
    formData.append("file", dataURLtoFile(sketchData, imageName));

    // Upload sketch to backend
    axios.post(`${API_URL}/upload_sketch/${imageName}`, formData)
      .then(() => alert("Sketch saved!"))
      .catch((error) => console.error("Upload failed", error));
  };

  // Convert base64 to File
  const dataURLtoFile = (dataUrl, filename) => {
    let arr = dataUrl.split(","), 
        mime = arr[0].match(/:(.*?);/)[1],
        bstr = atob(arr[1]),
        n = bstr.length, 
        u8arr = new Uint8Array(n);
    while (n--) {
      u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, { type: mime });
  };

  return (
    <div className="flex flex-col items-center bg-gray-100 min-h-screen p-6">
      <h1 className="text-3xl font-bold mb-4">Lego Image Sketcher</h1>

      {/* Image Selection */}
      <div className="mb-4">
        <h3 className="text-xl font-semibold">Select an Image:</h3>
        <div className="flex flex-wrap gap-2">
          {images.map((img, index) => (
            <button
              key={index}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-700"
              onClick={() => loadImage(img)}
            >
              {img}
            </button>
          ))}
        </div>
      </div>

      {/* Image & Sketchpad */}
      {selectedImage && (
        <div className="flex flex-col items-center mt-4">
          <h3 className="text-xl font-semibold mb-2">Image Preview</h3>
          <img src={imageSrc} alt="Selected" className="w-96 h-96 border-2 border-gray-400" />

          <h3 className="text-xl font-semibold mt-4 mb-2">Sketch Here:</h3>
          <ReactSketchCanvas
            ref={sketchRef}
            width="384px"
            height="384px"
            strokeWidth={4}
            strokeColor="black"
            backgroundImage={imageSrc}
            className="border-2 border-gray-400"
          />

          <button
            className="mt-4 bg-green-500 text-white px-6 py-2 rounded hover:bg-green-700"
            onClick={saveSketch}
          >
            Save Sketch
          </button>
        </div>
      )}
    </div>
  );
};

export default App;
