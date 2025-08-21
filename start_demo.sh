#!/bin/bash

echo "========================================"
echo "RAG Document Processing Utility - Demo"
echo "========================================"
echo ""
echo "Starting the demo system..."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python is not installed or not in PATH"
        echo "Please install Python 3.8+ and try again"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "ERROR: Python 3.8+ is required. Current version: $PYTHON_VERSION"
    exit 1
fi

echo "Python $PYTHON_VERSION found. Starting demo..."
echo ""

# Check if quick_start.py exists
if [ ! -f "quick_start.py" ]; then
    echo "ERROR: quick_start.py not found"
    echo "Please ensure you're in the correct directory"
    exit 1
fi

# Make the script executable
chmod +x quick_start.py

# Run the quick start script
$PYTHON_CMD quick_start.py

echo ""
echo "Demo completed."
