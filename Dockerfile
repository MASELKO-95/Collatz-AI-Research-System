# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-dev \
    build-essential \
    g++ \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for testing and development
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    pyyaml

# Copy source code
COPY . .

# Compile C++ modules
RUN cd src && \
    g++ -shared -fPIC -O3 -o libcollatz.so collatz_core.cpp && \
    g++ -shared -fPIC -O3 -std=c++11 -pthread -o libloop_searcher.so loop_searcher.cpp

# Create checkpoints directory
RUN mkdir -p checkpoints

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command
CMD ["python", "-m", "src.train"]

# Expose port for potential web interface (future feature)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1
