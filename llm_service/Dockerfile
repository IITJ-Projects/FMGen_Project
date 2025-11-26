# LLM Service Dockerfile - LLaMA-3-8B with vLLM
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV HF_HOME=/app/models
ENV TRANSFORMERS_CACHE=/app/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    git \
    build-essential \
    wget \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with specific versions for vLLM compatibility
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Create models directory
RUN mkdir -p /app/models

# Set proper permissions
RUN chmod +x main.py

# Expose port
EXPOSE 8002

# Health check with longer timeout for model loading
HEALTHCHECK --interval=60s --timeout=60s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Run the application
CMD ["python", "main.py"]