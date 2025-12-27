#!/bin/bash

# Start script for KNIT-LLM backend

echo "Starting KNIT-LLM Backend Server..."
echo ""

# Check if .env exists
if [ ! -f "../.env" ]; then
    echo "⚠️  Warning: .env file not found!"
    echo "Please create .env file from .env.example and add your GEMINI_API_KEY"
    echo ""
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
if [ ! -f "venv/.installed" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    python -m spacy download en_core_web_md
    touch venv/.installed
fi

# Start server
echo "Starting server..."
python main.py

