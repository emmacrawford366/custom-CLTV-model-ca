# Use an official Python base image
FROM python:3.9-slim

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy your training script
COPY train.py /opt/ml/code/train/train.py

# Set the entrypoint for training
ENTRYPOINT ["python", "/opt/ml/code/train/train.py"]
