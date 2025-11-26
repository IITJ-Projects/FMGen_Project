# Conversational AI Agent - Server Deployment Guide

## 🚀 Complete Deployment Process

This guide will walk you through deploying your Conversational AI Agent on a production server with GPU support.

## 📋 Prerequisites

### Server Requirements

- **OS**: Ubuntu 24.04 (recommended)
- **CPU**: 25+ vCPUs
- **RAM**: 64GB+ (32GB minimum)
- **GPU**: NVIDIA GPU with 24GB+ VRAM (NVIDIA L4 or better)
- **Storage**: 500GB+ SSD storage
- **Network**: Stable internet connection

### Software Requirements

- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **NVIDIA Docker**: nvidia-docker2
- **NVIDIA Drivers**: 550+ (for CUDA 12.6)

## 🔧 Step-by-Step Deployment

### Step 1: Server Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential tools
sudo apt install -y curl wget git htop nvtop
```

### Step 2: Install Docker and NVIDIA Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Docker (Official method)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
sudo apt install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Logout and login again for group changes
# Or run: newgrp docker
```

### Step 3: Install NVIDIA Drivers (if not already installed)

```bash
# Install NVIDIA drivers
sudo apt install -y nvidia-driver-550
sudo reboot

# After reboot, verify GPU access
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.6.1-base-ubuntu24.04 nvidia-smi
```

### Step 4: Clone and Setup Project

```bash
# Clone your repository
git clone https://github.com/YOUR_USERNAME/zevo-ai.git
cd zevo-ai

# Create necessary directories
mkdir -p logs config
mkdir -p data/{models,embeddings,reranker}

# Set proper permissions
sudo chown -R $USER:$USER .
chmod -R 755 .
```

### Step 5: Environment Configuration

```bash
# Create environment file
cat > .env << 'EOF'
# Model configurations
MODEL_NAME=faster-whisper-medium
LLM_MODEL_NAME=meta-llama/Llama-3-8B-Instruct
TTS_MODEL_NAME=melo-tts
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL_NAME=BAAI/bge-reranker-large

# GPU settings
DEVICE=cpu
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=4096
QUANTIZATION=awq

# Service URLs (internal Docker network)
ASR_SERVICE_URL=http://asr-service:8001
LLM_SERVICE_URL=http://llm-service:8002
RAG_SERVICE_URL=http://rag-service:8004
TTS_SERVICE_URL=http://tts-service:8003
QDRANT_HOST=qdrant-db
QDRANT_PORT=6333

# Logging
LOG_LEVEL=INFO

# TTS settings
VOICE_ID=default
STREAMING=true
SAMPLE_RATE=24000
EOF
```

### Step 6: Build Docker Images

```bash
# Build all services (this will take time for the first build)
echo "Building ASR Service..."
docker build -t zevo-ai/asr-service:latest -f asr_service/Dockerfile .

echo "Building TTS Service..."
docker build -t zevo-ai/tts-service:latest -f tts_service/Dockerfile .

echo "Building LLM Service..."
docker build -t zevo-ai/llm-service:latest -f llm_service/Dockerfile .

echo "Building RAG Service..."
docker build -t zevo-ai/rag-service:latest -f rag_service/Dockerfile .

echo "Building Orchestration Service..."
docker build -t zevo-ai/orchestration-service:latest -f orchestration_service/Dockerfile .

echo "All images built successfully!"
```

### Step 7: Deploy Services

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# Monitor logs
docker-compose logs -f
```

### Step 8: Verify Deployment

```bash
# Check all service health
curl http://localhost:8000/health

# Check individual services
curl http://localhost:8001/health  # ASR
curl http://localhost:8002/health  # LLM
curl http://localhost:8003/health  # TTS
curl http://localhost:8004/health  # RAG
curl http://localhost:6333/health  # Qdrant
```

### Step 9: Test the Pipeline

```bash
# Test conversation start
curl -X POST http://localhost:8000/conversation/start \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "language": "en"}'

# Test LLM generation
curl -X POST http://localhost:8002/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 100}'

# Test RAG retrieval
curl -X POST http://localhost:8004/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "artificial intelligence", "top_k": 5}'
```

## 🔒 Production Security Setup

### Step 10: Configure Firewall

```bash
# Install UFW
sudo apt install -y ufw

# Configure firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 8000/tcp  # Orchestration service
sudo ufw allow 8001/tcp  # ASR service
sudo ufw allow 8002/tcp  # LLM service
sudo ufw allow 8003/tcp  # TTS service
sudo ufw allow 8004/tcp  # RAG service
sudo ufw allow 6333/tcp  # Qdrant

# Enable firewall
sudo ufw enable
```

### Step 11: Setup SSL/TLS (Optional)

```bash
# Install Nginx
sudo apt install -y nginx certbot python3-certbot-nginx

# Create Nginx configuration
sudo tee /etc/nginx/sites-available/zevo-ai << 'EOF'
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/zevo-ai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

## 📊 Monitoring and Maintenance

### Step 12: Setup Monitoring

```bash
# Install monitoring tools
sudo apt install -y htop nvtop iotop

# Create monitoring script
cat > monitor.sh << 'EOF'
#!/bin/bash
echo "=== System Resources ==="
echo "CPU Usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1
echo "Memory Usage:"
free -h | grep Mem | awk '{print $3"/"$2}'
echo "GPU Usage:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
echo "=== Docker Services ==="
docker-compose ps
echo "=== Service Health ==="
curl -s http://localhost:8000/health | jq .
EOF

chmod +x monitor.sh
```

### Step 13: Setup Log Rotation

```bash
# Create log rotation configuration
sudo tee /etc/logrotate.d/zevo-ai << 'EOF'
/Users/apple/Documents/ZevoCode/zevo-ai/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 root root
}
EOF
```

## 🔄 Update and Maintenance

### Update Services

```bash
# Pull latest code
git pull origin main

# Rebuild and restart services
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

### Backup and Recovery

```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="./backups/$DATE"
mkdir -p $BACKUP_DIR

# Backup volumes
docker run --rm -v zevo-ai_qdrant_data:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/qdrant_data.tar.gz -C /data .
docker run --rm -v zevo-ai_llm_model_cache:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/llm_models.tar.gz -C /data .

# Backup configuration
cp docker-compose.yml $BACKUP_DIR/
cp .env $BACKUP_DIR/

echo "Backup completed: $BACKUP_DIR"
EOF

chmod +x backup.sh
```

## 🚨 Troubleshooting

### Common Issues

1. **GPU Not Available**

   ```bash
   # Check NVIDIA Docker
   docker run --rm --gpus all nvidia/cuda:12.6.1-base-ubuntu24.04 nvidia-smi

   # Check Docker Compose GPU syntax
   docker-compose config
   ```

2. **Service Health Check Failures**

   ```bash
   # Check service logs
   docker-compose logs <service-name>

   # Restart specific service
   docker-compose restart <service-name>
   ```

3. **Out of Memory**

   ```bash
   # Check memory usage
   free -h

   # Adjust GPU memory utilization in docker-compose.yml
   # Change GPU_MEMORY_UTILIZATION to 0.8 or lower
   ```

4. **Model Download Issues**

   ```bash
   # Check internet connectivity
   curl -I https://huggingface.co

   # Clear Docker cache
   docker system prune -a
   ```

## 📈 Performance Optimization

### GPU Optimization

- Adjust `GPU_MEMORY_UTILIZATION` based on your GPU
- Use AWQ quantization for LLaMA-3-8B
- Monitor GPU usage with `nvidia-smi`

### Memory Optimization

- Set appropriate `MAX_MODEL_LEN` for your use case
- Use model caching effectively
- Monitor memory usage with `htop`

### Network Optimization

- Use internal Docker network for service communication
- Implement connection pooling
- Monitor network usage with `iotop`

## ✅ Deployment Checklist

- [ ] Server meets hardware requirements
- [ ] Docker and NVIDIA Docker installed
- [ ] NVIDIA drivers installed and working
- [ ] Project cloned and directories created
- [ ] Environment variables configured
- [ ] Docker images built successfully
- [ ] Services started and healthy
- [ ] Firewall configured
- [ ] SSL/TLS configured (if needed)
- [ ] Monitoring setup
- [ ] Backup strategy implemented
- [ ] Test pipeline working

## 🎯 Next Steps

1. **Load your data** into the RAG service
2. **Fine-tune models** for your specific domain
3. **Implement authentication** for production use
4. **Add rate limiting** and API key management
5. **Setup alerting** for service health
6. **Implement CI/CD** for automated deployments

Your Conversational AI Agent is now ready for production use! 🚀
