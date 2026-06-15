# Use a stable slim base image (better compatibility)
FROM python:3.11-slim-bookworm

# Prevent Python from writing .pyc files + improve logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies (important for ML libs like pandas, sklearn, xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip for faster + stable installs
RUN pip install --upgrade pip setuptools wheel

# Copy only requirements first (maximizes Docker layer caching)
COPY requirements-prod.txt .

# Install dependencies efficiently
RUN pip install --no-cache-dir --prefer-binary -r requirements-prod.txt

# Copy application code LAST (avoids breaking cache)
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with production-ready settings
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]