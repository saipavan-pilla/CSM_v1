# Use the Node.js base image for the frontend
FROM node:latest AS frontend

# Set the working directory for the frontend
WORKDIR /app/frontend

# Copy the frontend application code
COPY frontend/ /app/frontend

# Install frontend dependencies and build the React.js application
RUN npm install
RUN npm run build
