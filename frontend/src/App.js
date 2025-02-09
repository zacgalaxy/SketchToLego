import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import { ReactSketchCanvas } from "react-sketch-canvas";

//const API_URL = "https://0zn01bzj-8000.uks1.devtunnels.ms"; // FastAPI backend URL
const API_URL = "http://localhost:8000";
const Home = () => {
  const [images, setImages] = useState([]); // List of images
  const [selectedImage, setSelectedImage] = useState(null); // Current image
  const [imageSrc, setImageSrc] = useState(""); // Image URL for viewing
  const sketchRef = useRef(null); // Reference to sketchpad
  const [isEraser, setIsEraser] = useState(false); // Eraser mode
  const [sketches, setSketches] = useState([]); // Store fetched sketches
  const [showPopup, setShowPopup] = useState(false);
  
  // Fetch list of images from FastAPI
  useEffect(() => {
    axios.get(`${API_URL}/images`).then((response) => {
      setImages(response.data.images);
    });
  }, []);

  // Fetch list of sketch data from FastAPI
  useEffect(() => {
    axios.get(`${API_URL}/sketchesnum`)
      .then((response) => {
        setSketches(response.data.sketches); // 🔹 Corrected API response key
      })
      .catch((error) => console.error("Error fetching sketches:", error));
  }, []);
    


  // Load selected image
  const loadImage = (imageName) => {
    setSelectedImage(imageName);
    setImageSrc(`${API_URL}/image/${imageName}`);

    if (sketchRef.current) {
      sketchRef.current.clearCanvas();
  }
  };
  // add white background
  const whitebackgroundAndResize = (sketchData) => {
    return new Promise((resolve) => {
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");

        // Set final image size to 256x256
        canvas.width = 256;
        canvas.height = 256;

        // Fill the background with white
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Load sketch (transparent PNG, 512x512)
        const sketchImg = new Image();
        sketchImg.src = sketchData;
        sketchImg.onload = () => {
            // Resize & draw sketch onto the 256x256 white background
            ctx.drawImage(sketchImg, 0, 0, 256, 256);
            resolve(canvas.toDataURL("image/png")); // Return resized image
        };
    });
};



  // Save the sketch
  const saveSketch = async () => {
    if (!sketchRef.current || !selectedImage) return;

    try{
    // Convert sketch to PNG format
    const sketchData = await sketchRef.current.exportImage("");
    
    // Convert base64 to image and merge with white background
    const finalImage = await whitebackgroundAndResize(sketchData);

    const imageName = selectedImage; // Save with same name as the image

    // Convert base64 data to File object
    const formData = new FormData();
    formData.append("file", dataURLtoFile(finalImage, imageName));

    // Upload sketch to backend
    axios.post(`${API_URL}/upload_sketch/${imageName}`, formData)
      .then(() => alert("Sketch saved!"))
      .catch((error) => console.error("Upload failed", error));
    }catch (error) {
      console.error("Error saving sketch:", error);
  }
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

        <div className="container">
              <header>
                <img src="/crocodile.jpg" alt="Crocodile Logo" width="150" />
              </header>
            <h1>Lego Image Sketcher
            <div className="Top-buttons">
                  <Link to="/lego-generator">
                      <button key="lego Generator">Lego generator</button>
                  </Link>
                  <button onClick={() => setShowPopup(true)}>Instructions!</button>
              </div>
            </h1>
            {showPopup && (
                <div className="popup-overlay">
                  <div className="popup-box">
                    <h2>Instructions</h2>
                    <p>1. Click an image to load it.</p>
                    <p>2. Sketch image in same orientation</p>
                    <p>3. Does not have to be exact, these are rough sketches</p>
                    <p>4. Use the eraser to fix mistakes.</p>
                    <p>5. Click "Save Sketch" to save your work.</p>
                    <button onClick={() => setShowPopup(false)}>Close</button>
                  </div>
                </div>
    )}

            {/* Button Grid (10x5) */}
            <div className="button-grid">
                {images.slice(0, 50).map((img, index) => {
                  const imageName = img.replace(/\.(jpeg|jpg|png)$/i, ""); // Remove file extension

                  // Find matching sketch count
                  const matchingSketch = sketches.find(sketch => sketch.folder === imageName);
                  const sketchCount = matchingSketch ? matchingSketch.num_files : 0; // Default to 0 if not found

                  return (
                    <button key={index} onClick={ () => sketchCount<5 ?  loadImage(img): null } disabled= {sketchCount>= 5}>
                      {imageName} ({sketchCount}) {/* Show image name and sketch count */}
                    </button>
                  );
                })}
              </div>



            {/* Centered Image */}
            {selectedImage && (
                <div className="image-sketch-container">
                    {/* Image on Left */}
                    <div className="image-box">
                        <h3>{selectedImage.replace(/\.(jpeg|jpg|png)$/i, "")}</h3>
                        <img src={imageSrc} alt="Selected" />
                    </div>

                    {/* Sketchpad Below */}
                     <div className="sketch-box">
                        <h3>Sketch Here</h3>
                        <ReactSketchCanvas
                            ref={sketchRef}
                            width="100%"
                            height="100%"
                            strokeWidth={isEraser ? 15 : 6} // Increase stroke width for eraser
                            strokeColor={isEraser ? "white" : "black"} // White for erasing
                            className="sketch-container"
                        />
                    </div>
                    </div>
                
            )}

            {/* Eraser & Save Buttons */}
                    <div className="action-buttons">
                        <button onClick={() => setIsEraser(!isEraser)}>
                            {isEraser ? "Switch to Draw" : "Switch to Eraser"}
                        </button>
                        <button onClick={saveSketch}>Save Sketch</button>
                        <button onClick={() => sketchRef.current.clearCanvas()}>Clear Sketch</button>
                        
             </div>  
        </div>
    );

};

// Second Page (Lego Generator)
const LegoGenerator = () => {
  return (
      <div className="container">
          <h1>Lego Generator Page</h1>
          <p>This is where the Lego Generator feature will go.</p>
          <Link to="/">
              <button>Back to Home</button>
          </Link>
      </div>
  );
};

// Main App Component with Routing
const App = () => {
  return (
      <Router>
          <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/lego-generator" element={<LegoGenerator />} />
          </Routes>
      </Router>
  );
};

export default App;


