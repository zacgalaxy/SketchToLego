import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import { ReactSketchCanvas } from "react-sketch-canvas";

//const API_URL = "https://0zn01bzj-8000.uks1.devtunnels.ms"; // FastAPI backend URL
const API_URL = "http://localhost:8000";
const Home = () => {
  const [images, setImages] = useState([]); // List of images
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedfile, setFilename] = useState(null); // Current image
  const [imageSrc, setImageSrc] = useState(""); // Image URL for viewing
  const sketchRef = useRef(null); // Reference to sketchpad
  const [isEraser, setIsEraser] = useState(false); // Eraser mode
  const [sketches, setSketches] = useState([]); // Store fetched sketches
  const [showPopup, setShowPopup] = useState(false);
  
  // Fetch list of images from FastAPI
  useEffect(() => {
    axios.get(`${API_URL}/images`).then((response) => {
      setImages(response.data);
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
  const loadImage = (filename, displayname) => {
    setSelectedImage(displayname);
    setFilename(filename)
    setImageSrc(`${API_URL}/image/${filename}`);

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



  //analize sketch
  const analyisesketch= async(sketch) => {
    if (!selectedImage) return null;
    try {
      console.log("Checking sketch score for:", selectedfile);

      // Convert sketch data to File
      const formData = new FormData();
      formData.append("sketch", dataURLtoFile(sketch, "sketch.png"));

      // Call FastAPI for sketch similarity
      const response = await axios.post(`${API_URL}/analyze_sketch/${selectedfile}`, formData);
      alert(String(response));
      return response.data.final_hybrid_score; // Get similarity score
    } catch (error) {
      console.error("Error fetching sketch score:", error);
      return null;
    }
      };



  // Save the sketch
  const saveSketch = async () => {
    if (!sketchRef.current || !selectedImage || !selectedfile ) return;

    try{
    // Convert sketch to PNG format
    const sketchData = await sketchRef.current.exportImage("");
    
    // Convert base64 to image and merge with white background
    const finalImage = await whitebackgroundAndResize(sketchData);

    
    //get sim score
    const sketchScore= await analyisesketch(finalImage);
    if (sketchScore===null){
      alert ('error in analysing sketch please try again');
    }

    //Compaore sim score 
    if (sketchScore<5){
      alert('Sketch score of '+String(sketchScore)+' does not match well enough try again');
      sketchRef.current.clearCanvas();
      return;
    }
    const imageName = selectedfile; // ✅ Use actual filename
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
  sketchRef.current.clearCanvas();
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
                  const { filename, display_name } = img; // ✅ Extract filename & display name

                  // Find matching sketch count
                  const matchingSketch = sketches.find(sketch => sketch.folder === filename.replace(/\.(jpeg|jpg|png)$/i, ""));
                  const sketchCount = matchingSketch ? matchingSketch.num_files : 0;

                  return (
                    <button key={index} onClick={() => sketchCount < 5 ? loadImage(filename, display_name) : null} disabled={sketchCount >= 5}>
                      {display_name} ({sketchCount}) {/* ✅ Uses display name */}
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


