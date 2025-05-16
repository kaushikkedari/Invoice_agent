# Use Python 3.12 slim Bullseye as base image
FROM python:3.11-slim-bullseye
 
WORKDIR /app
 
# Install Docker CLI and dependencies
RUN apt-get update && apt-get install -y default-jre\
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y docker-ce-cli containerd.io \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
 
# Copy requirements file
COPY requirements.txt /app
 
# Create and activate virtual environment
RUN python -m venv /opt/venv
 
# Update pip and install dependencies
RUN /opt/venv/bin/pip install --upgrade pip && \
/opt/venv/bin/pip install -r requirements.txt
 
# Copy application code and credentials
COPY . /app
 
# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
 
# Expose port
EXPOSE 8000
 
# Start the application
ENTRYPOINT ["/bin/sh", "-c", "/opt/venv/bin/uvicorn pipeline_ai:app --workers ${FUNCTIONS_WORKER_PROCESS_COUNT:-10} --host 0.0.0.0 --port 8080 --loop asyncio --timeout-keep-alive 6000 --log-level debug"]
