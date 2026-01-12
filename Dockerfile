# Use a base image with PyTorch and CUDA pre-installed
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (GL, GIT, etc.)
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    libxi6 \
    libxrender1 \
    libxxf86vm1 \
    libxfixes3 \
    libxcursor1 \
    libxinerama1 \
    libxrandr2 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /code

# Install Python dependencies
# 1. Base requirements (Graduo, Spaces, REMBG, etc)
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt

# 2. Pytorch3D (Requires Torch to be present)
# Using a specific commit or tag can be safer, but for now we follow the "latest" approach or a known working logic.
# The base image has torch.
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/pytorch3d.git

# 3. Kaolin (Optional/Specific to SAM3D?)
# SAM3D docs say: export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
# But we are on Torch 2.1.2. We should try to find a compatible Kaolin or build from source.
# Let's try standard pip install first, if it fails we might need specific wheels.
# Using --no-deps for Kaolin to avoid it messing up Torch if versions mismatch?
RUN pip install --no-cache-dir kaolin==0.16.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.2_cu121.html || pip install kaolin

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

# Command to run the application
CMD ["python", "app.py"]
