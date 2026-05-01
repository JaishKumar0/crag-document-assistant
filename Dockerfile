# Use a lightweight Python 3.13 base image
FROM python:3.13-slim

# Set the working directory
WORKDIR /project

# Install system dependencies (Crucial for OpenCV and PyMuPDF)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt streamlit requests

# Copy the rest of the application code
COPY . .

# Expose both ports
EXPOSE 8000
EXPOSE 8501

# Start BOTH the FastAPI backend (in the background using &) AND Streamlit
CMD uvicorn app.main:api --host 0.0.0.0 --port 8000 & streamlit run app/frontend.py --server.port=8501 --server.address=0.0.0.0