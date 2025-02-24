#!/bin/bash
# Exit immediately if a command exits with a non-zero status.

#chmod +x setup_and_run.sh
##before running    

set -e

echo "Creating Python virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Python dependencies..."
pip install -r requirements.txt

# Start the FastAPI backend in the background.
echo "Starting FastAPI backend..."


# Change directory to the React app folder.
echo "Setting up React frontend..."
cd fastapi-react-ui

# Install Node.js dependencies if package.json exists.
if [ -f package.json ]; then
    npm install
    echo "Starting React development server..."
    # npm start  
    # run this after the script
else
    echo "No package.json found in fastapi-react-ui. Skipping React setup."
fi
cd ..
uvicorn app:app --reload --host 0.0.0.0 --port 8081
echo "backend is running.Run the front end"
echo "Press Ctrl+C to stop."
