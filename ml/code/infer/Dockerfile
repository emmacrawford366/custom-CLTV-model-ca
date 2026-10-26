# Use an official Python base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements.txt into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip

# Install the required Python packages via pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training and inference scripts inside the container
COPY infer.py /opt/ml/code/infer/infer.py

# Set environment variables
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code/infer
ENV SAGEMAKER_PROGRAM=infer.py

# Define the entry point script
ENTRYPOINT ["python", "/opt/ml/code/infer/infer.py"]

