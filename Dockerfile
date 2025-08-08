# Use a minimal base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies including git and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code into the container
COPY ./app ./app

# Expose the port your app will run on
EXPOSE 8080

# Command to run your FastAPI app with uvicorn on port 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
