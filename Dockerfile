# Dockerfile

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install build dependencies for compiling pandas, xgboost if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app ./app
COPY tests ./tests

# Expose port for Cloud Run
EXPOSE 8080

# Run FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
