# Use minimal official Python 3.10 slim image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies for building and running (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (your app directory and other files)
COPY ./app ./app
COPY main.py .
COPY train.py .
COPY README.md .

# Expose port 8080 (Cloud Run standard)
EXPOSE 8080

# Run the app with uvicorn on port 8080, binding to 0.0.0.0 to allow outside access
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
