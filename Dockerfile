FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by OpenCV and ML libs on headless servers
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

# Copy full project
COPY . .

# Hugging Face Spaces uses port 7860
EXPOSE 7860

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
