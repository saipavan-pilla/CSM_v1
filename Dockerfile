# Use the Node.js base image for the frontend
FROM node:latest AS frontend

# Set the working directory for the frontend
WORKDIR /app/frontend

# Copy the frontend application code
COPY frontend/ /app/frontend

# Install frontend dependencies and build the React.js application
RUN npm install
RUN npm run build

# Switch to a Python base image for the backend
FROM python:3.11 AS backend

# Set the working directory for the backend
WORKDIR /app/backend

# Copy the backend application code
COPY backend/ /app/backend

# Install backend Python dependencies
RUN pip install -r requirements.txt
RUN pip install -r requirements1.txt
RUN pip install -r requirements2.txt
RUN pip install -r requirements3.txt

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev
    

# Expose the necessary port for the backend
EXPOSE 8000

# Set the command to run the backend server
CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
