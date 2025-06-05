FROM ubuntu:22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy the app and dependencies
COPY requirements.txt .
COPY src/ ./src/
COPY data/ ./data/
COPY start.sh .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Make the script executable
RUN chmod +x start.sh

# Set the command to run the application
CMD ["./start.sh"]