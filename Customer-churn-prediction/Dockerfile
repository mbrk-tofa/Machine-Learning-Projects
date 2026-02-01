# -------------------------
# Base image
# -------------------------
FROM python:3.10-slim

# -------------------------
# Environment settings
# -------------------------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# -------------------------
# Set working directory
# -------------------------
WORKDIR /app

# -------------------------
# Install system dependencies
# -------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# Copy dependency list
# -------------------------
COPY requirements.txt .

# -------------------------
# Install Python dependencies
# -------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------
# Copy project files
# -------------------------
COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY data/ data/

# -------------------------
# Expose API port
# -------------------------
EXPOSE 8000

# -------------------------
# Run FastAPI with Uvicorn
# -------------------------
CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
