#!/bin/bash

# Start backend server script

cd "$(dirname "$0")"

echo "=========================================="
echo "Starting KNIT-LLM Backend Server"
echo "=========================================="
echo ""

# Check if .env exists in parent directory
if [ ! -f "../.env" ]; then
    echo "❌ ERROR: .env file not found in project root!"
    echo "   Please create .env file with your GEMINI_API_KEY"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: python3 not found"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
python3 -c "import fastapi" 2>/dev/null || {
    echo "❌ FastAPI not installed. Installing dependencies..."
    python3 -m pip install fastapi 'uvicorn[standard]' python-dotenv google-generativeai sentence-transformers networkx scikit-learn requests pydantic python-multipart --user
}

echo ""
echo "Starting server..."
echo "Backend will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""
echo "=========================================="
echo ""

# Start the server
python3 main.py

