FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY main.py .
COPY s3_utils.py .
COPY chainlit.md .
COPY chainlit.yaml .
COPY README.md .

# Create necessary directories and set permissions
RUN mkdir -p public && \
    mkdir -p /app/.files && \
    chmod -R 777 /app/.files && \
    chmod -R 777 /app

# Create a non-root user
RUN useradd -m -u 1000 user

# Change ownership of the app directory to the non-root user
RUN chown -R user:user /app

# Switch to the non-root user
USER user

# Set AWS configuration for anonymous access
ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""
ENV AWS_REGION="us-east-1"

# Hugging Face Spaces specific environment variable
ENV HOME=/app

# This will be overridden by Hugging Face Spaces secrets
ENV OPENAI_API_KEY=""

# Enable Python unbuffered output for better logging
ENV PYTHONUNBUFFERED=1

# Expose port required by Hugging Face Spaces
EXPOSE 7860

# Command to run the application with more verbose logging
CMD ["chainlit", "run", "main.py", "--host", "0.0.0.0", "--port", "7860", "--debug"]