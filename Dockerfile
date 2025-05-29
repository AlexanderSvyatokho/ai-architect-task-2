# Use official Ollama image
FROM ollama/ollama:latest

# Set working directory
WORKDIR /app

# Copy the app and dependencies
COPY requirements.txt .
COPY src/ ./src/
COPY start.sh .

# Install Python + dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Make the script executable
RUN chmod +x start.sh

# Override default ENTRYPOINT from ollama/ollama
ENTRYPOINT []

# Set the command to run the application
CMD ["./start.sh"]
