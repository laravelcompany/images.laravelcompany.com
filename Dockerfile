# Use a slimmer base image for CPU-only deployment
FROM python:3.11-slim AS final

# Set working directory
WORKDIR /home/trending

# Add labels for better maintainability
LABEL maintainer="Stefan Bogdanel <stefan@izdrail.com>"
LABEL organization="Laravel Company"

# Set environment variables in a single layer to reduce image size
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CLOUDFLARE_API_TOKEN="" \
    CLOUDFLARE_ACCOUNT_ID="" \
    CUDA_VISIBLE_DEVICES="" \
    TORCH_CUDA_AVAILABLE=0 \
    HF_HOME=/home/trending/cache \
    TRANSFORMERS_CACHE=/home/trending/cache \
    HF_DATASETS_CACHE=/home/trending/cache \
    HF_HUB_CACHE=/home/trending/cache

# Install system dependencies in a single RUN to minimize layers
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    mlocate \
    net-tools \
    software-properties-common \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in a single layer
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir supervisor pipx

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU version
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install additional packages for image-to-image API
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    pydantic \
    requests \
    diffusers \
    transformers \
    accelerate \
    pillow \
    python-multipart \
    safetensors

# Create cache directory with proper permissions
RUN mkdir -p /home/trending/cache && chmod 755 /home/trending/cache

# Create and run model download script
RUN echo '#!/usr/bin/env python3\n\
import os\n\
import torch\n\
from diffusers import StableDiffusionImg2ImgPipeline\n\
import logging\n\
\n\
logging.basicConfig(level=logging.INFO)\n\
logger = logging.getLogger(__name__)\n\
\n\
def download_model():\n\
    try:\n\
        os.environ["CUDA_VISIBLE_DEVICES"] = ""\n\
        os.environ["TORCH_CUDA_AVAILABLE"] = "0"\n\
        torch.cuda.is_available = lambda: False\n\
        model_id = "runwayml/stable-diffusion-v1-5"\n\
        cache_dir = "/home/trending/cache"\n\
        logger.info(f"Downloading model {model_id} to {cache_dir}...")\n\
        pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(\n\
            model_id,\n\
            torch_dtype=torch.float32,\n\
            cache_dir=cache_dir,\n\
            safety_checker=None,\n\
            requires_safety_checker=False,\n\
            local_files_only=False\n\
        )\n\
        logger.info("Model downloaded successfully!")\n\
        pipeline = pipeline.to("cpu")\n\
        logger.info("Model test successful - ready for offline use!")\n\
    except Exception as e:\n\
        logger.error(f"Error downloading model: {str(e)}")\n\
        raise e\n\
\n\
if __name__ == "__main__":\n\
    download_model()\n\
' > /tmp/download_model.py && \
    chmod +x /tmp/download_model.py && \
    python3 /tmp/download_model.py && \
    rm /tmp/download_model.py

# Install Zsh with spaceship prompt and plugins
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t https://github.com/denysdovhan/spaceship-prompt \
    -a 'SPACESHIP_PROMPT_ADD_NEWLINE="false"' \
    -a 'SPACESHIP_PROMPT_SEPARATE_LINE="false"' \
    -p git \
    -p ssh-agent \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions

# Copy application code
COPY . .

# Update database for mlocate
RUN updatedb

# Copy supervisord configuration
COPY docker/supervisord.conf /etc/supervisord.conf

# Expose ports
EXPOSE 1054 8000

# Health check for the image-to-image API
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:1054/health || exit 1

# Run application with supervisord
ENTRYPOINT ["supervisord", "-c", "/etc/supervisord.conf", "-n"]