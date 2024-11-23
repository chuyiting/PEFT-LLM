# Start from the PyTorch base image with CUDA support
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

# Update system packages and install required dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Ensure PyTorch, numpy, pandas, and CUDA are installed
RUN pip install --no-cache-dir --upgrade \
    numpy \
    pandas

# Install the required Python packages
RUN pip install --no-cache-dir --upgrade \
    transformers \
    bitsandbytes \
    peft \
    accelerate \
    datasets \
    trl

# Set default command to python3
CMD ["python3"]
