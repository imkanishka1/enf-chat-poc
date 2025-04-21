FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for chromadb/onnxruntime
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create a directory for PDFs (in case none are provided)
RUN mkdir -p ./pdfs

# Verify app.py exists
RUN test -f app.py || (echo "app.py not found" && exit 1)

# Expose port 80
EXPOSE 80

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=80
ENV PYTHONPATH=/app

# Use python -m flask run for reliability
CMD ["python", "-m", "flask", "run"]