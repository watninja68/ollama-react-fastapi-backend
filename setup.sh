#!/bin/bash
# Exit immediately if a command exits with a non-zero status.

#chmod +x setup_and_run.sh
##before running    

set -e

echo "Creating Python virtual environment..."
python3.11.4 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Python dependencies..."
pip install -r requirements.txt

# Start the FastAPI backend in the background.

uvicorn app:app --reload --host 0.0.0.0 --port 8081
echo "backend is running.Run the front end"
echo "Press Ctrl+C to stop."
