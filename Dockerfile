# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port 8081 to the outside world
EXPOSE 8081

# Optional: if your application needs to refer to the host’s Ollama service,
# set an environment variable to point to it. The following assumes the Ollama service
# is available at port 11434 on your host.
ENV OLLAMA_HOST=http://host.docker.internal:11434

# Command to run the application using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8081"]
# docker run --network host docproc-app
