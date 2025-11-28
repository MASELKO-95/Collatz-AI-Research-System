#!/bin/bash
set -e

# Ensure we are in the script's directory or handle paths correctly
# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo "Setting up environment in $SCRIPT_DIR..."

# Create virtual environment if it doesn't exist
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running Training..."
python3 -m src.train

echo "Running Analysis..."
python3 -m src.analyze
