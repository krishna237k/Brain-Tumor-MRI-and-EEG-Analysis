FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and assets
COPY app.py .
COPY mri_eeg_core.py .
COPY models/ models/
COPY sample_data/ sample_data/
COPY .streamlit/ .streamlit/

# Default port (fallback to 8501 if PORT env is not specified by platform)
ENV PORT=8501
EXPOSE 8501

# Health check to ensure the container is responsive
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:${PORT}/_stcore/health || exit 1

# Launch Streamlit app with dynamic port support
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
