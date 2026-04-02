FROM python:3.11-slim

# Install system libraries MediaPipe needs
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgles2-mesa \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install Python libraries
RUN pip install --no-cache-dir \
    flask \
    opencv-python-headless \
    mediapipe \
    scikit-learn \
    numpy \
    gunicorn

# Expose port 7860 (Hugging Face uses this port)
EXPOSE 7860

# Start the app
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]