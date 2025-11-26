#!/bin/bash

# Conversational AI Agent - Automated Deployment Script
# This script automates the deployment process for your server

set -e  # Exit on any error

echo "🚀 Starting Conversational AI Agent Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Check system requirements
print_status "Checking system requirements..."

# Check OS
if [[ ! -f /etc/os-release ]]; then
    print_error "Cannot determine OS"
    exit 1
fi

source /etc/os-release
if [[ "$ID" != "ubuntu" ]]; then
    print_warning "This script is optimized for Ubuntu. You're running $ID"
fi

# Check available memory
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
if [[ $TOTAL_MEM -lt 32 ]]; then
    print_warning "Recommended minimum RAM is 32GB. You have ${TOTAL_MEM}GB"
fi

# Check available disk space
DISK_SPACE=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
if [[ $DISK_SPACE -lt 100 ]]; then
    print_warning "Recommended minimum disk space is 100GB. You have ${DISK_SPACE}GB available"
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_status "Docker not found. Installing Docker..."
    
    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    
    print_success "Docker installed successfully"
else
    print_success "Docker is already installed"
fi

# Check if NVIDIA Docker is installed
if ! docker run --rm --gpus all nvidia/cuda:12.6.1-base-ubuntu24.04 nvidia-smi &> /dev/null; then
    print_status "NVIDIA Docker not found. Installing NVIDIA Container Toolkit..."
    
    # Install NVIDIA Container Toolkit
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
    
    print_success "NVIDIA Container Toolkit installed successfully"
else
    print_success "NVIDIA Docker is already working"
fi

# Check if project directory exists
if [[ ! -d "zevo-ai" ]]; then
    print_status "Project directory not found. Please clone your repository first:"
    echo "git clone https://github.com/YOUR_USERNAME/zevo-ai.git"
    echo "cd zevo-ai"
    exit 1
fi

# Create necessary directories
print_status "Creating necessary directories..."
mkdir -p logs config
mkdir -p data/{models,embeddings,reranker}

# Set proper permissions
sudo chown -R $USER:$USER .
chmod -R 755 .

# Create environment file if it doesn't exist
if [[ ! -f ".env" ]]; then
    print_status "Creating environment configuration..."
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
    print_success "Environment file created"
else
    print_success "Environment file already exists"
fi

# Check if Dockerfiles exist
SERVICES=("asr_service" "tts_service" "llm_service" "rag_service" "orchestration_service")
for service in "${SERVICES[@]}"; do
    if [[ ! -f "$service/Dockerfile" ]]; then
        print_error "Dockerfile not found for $service"
        print_error "Please create the Dockerfiles first"
        exit 1
    fi
done

# Build Docker images
print_status "Building Docker images (this may take a while)..."
for service in "${SERVICES[@]}"; do
    print_status "Building $service..."
    docker build -t zevo-ai/${service}:latest -f ${service}/Dockerfile .
    print_success "$service built successfully"
done

# Start services
print_status "Starting services..."
docker-compose up -d

# Wait for services to start
print_status "Waiting for services to start..."
sleep 30

# Check service health
print_status "Checking service health..."
HEALTH_CHECK_COUNT=0
MAX_HEALTH_CHECKS=10

while [[ $HEALTH_CHECK_COUNT -lt $MAX_HEALTH_CHECKS ]]; do
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "All services are healthy!"
        break
    else
        print_warning "Services not ready yet. Waiting..."
        sleep 30
        HEALTH_CHECK_COUNT=$((HEALTH_CHECK_COUNT + 1))
    fi
done

if [[ $HEALTH_CHECK_COUNT -eq $MAX_HEALTH_CHECKS ]]; then
    print_error "Services failed to start properly"
    print_status "Checking service logs..."
    docker-compose logs
    exit 1
fi

# Test the pipeline
print_status "Testing the pipeline..."
if curl -s -X POST http://localhost:8000/conversation/start \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "language": "en"}' > /dev/null; then
    print_success "Pipeline test successful"
else
    print_warning "Pipeline test failed"
fi

# Create monitoring script
print_status "Creating monitoring script..."
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

# Create backup script
print_status "Creating backup script..."
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

# Final status
print_success "🎉 Deployment completed successfully!"
echo ""
echo "📊 Service Status:"
docker-compose ps
echo ""
echo "🔗 Service Endpoints:"
echo "  - Orchestration: http://localhost:8000"
echo "  - ASR Service:   http://localhost:8001"
echo "  - LLM Service:   http://localhost:8002"
echo "  - TTS Service:   http://localhost:8003"
echo "  - RAG Service:   http://localhost:8004"
echo "  - Qdrant DB:     http://localhost:6333"
echo ""
echo "📋 Useful Commands:"
echo "  - Monitor services: ./monitor.sh"
echo "  - View logs: docker-compose logs -f"
echo "  - Backup data: ./backup.sh"
echo "  - Stop services: docker-compose down"
echo "  - Restart services: docker-compose restart"
echo ""
echo "🚀 Your Conversational AI Agent is ready!"
