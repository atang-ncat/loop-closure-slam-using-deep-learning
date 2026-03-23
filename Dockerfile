# ============================================================
# Loop Closure SLAM — GPU-accelerated training container
# ============================================================
# Base: NVIDIA PyTorch NGC container (CUDA 12.1 + PyTorch 2.1)
# ============================================================

FROM nvcr.io/nvidia/pytorch:23.10-py3

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/loop-closure-slam

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configs
COPY src/ src/
COPY configs/ configs/

# Default: train the model
CMD ["python3", "-m", "src.step6_train", "--config", "configs/config.yaml"]
