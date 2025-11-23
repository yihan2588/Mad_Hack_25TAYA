# Use a PyTorch image with CUDA support for Azure GPU
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies (git for cloning, ffmpeg for audio)
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone the MUSC Violin repo
RUN git clone https://github.com/MTG/violin-transcription.git MUSC_violin

# Copy your backend code
COPY main.py .

# Expose port
EXPOSE 8000

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
