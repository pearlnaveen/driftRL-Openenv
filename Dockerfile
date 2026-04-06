# Use lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create model directory (in case it doesn't exist)
RUN mkdir -p model

# Expose API port
EXPOSE 7860

# Run training first, then start API
CMD ["sh", "-c", "python train.py && uvicorn app:app --host 0.0.0.0 --port 8000"]