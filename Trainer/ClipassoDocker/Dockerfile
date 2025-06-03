# Use NVIDIA CUDA image with PyTorch support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# Prevent tzdata from prompting for user input
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and required libraries
RUN apt update && apt install -y \
    python3 python3-pip python3-venv git cmake \
    libgl1-mesa-glx libglib2.0-0 libosmesa6 libegl1-mesa \
    libxrandr2 libxinerama1 libxcursor1 \
    tzdata && ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && dpkg-reconfigure -f noninteractive tzdata

# Set working directory inside Docker
WORKDIR /workspace

# Copy necessary folders (clipasso & diffvg)
COPY clipasso /workspace/clipasso
COPY diffvg /workspace/diffvg

# Install latest PyTorch with CUDA 12.1
RUN pip3 install --upgrade pip
# Install Python dependencies for CLIPasso
WORKDIR /workspace/clipasso
RUN pip3 install --no-cache-dir -r requirements.txt ipywidgets protobuf==3.20.*
RUN pip3 uninstall torch torchvision torchaudio -y
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


# Install diffvg
WORKDIR /workspace
RUN rm -rf diffvg
RUN pip3 uninstall -y pydiffvg
RUN git clone --recursive https://github.com/BachiLi/diffvg.git /workspace/diffvg
WORKDIR /workspace/diffvg
RUN git submodule update --init --recursive
RUN python3 setup.py install

# Set back to main working directory
WORKDIR /workspace/clipasso

# Ensure script is executable
RUN chmod +x /workspace/clipasso/process_images.py

# Run the image processing script when the container starts
CMD ["python3", "/workspace/clipasso/process_images.py"]
