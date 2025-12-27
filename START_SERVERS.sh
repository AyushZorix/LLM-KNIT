#!/bin/bash

# Start script for both frontend and backend

echo "=========================================="
echo "KNIT-LLM Server Startup"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "Creating .env from .env.example..."
    cp .env.example .env
    echo ""
    echo "❌ IMPORTANT: Please edit .env and add your GEMINI_API_KEY"
    echo "   Then run this script again."
    exit 1
fi

# Check if API key is set
source .env 2>/dev/null || true
if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "your_gemini_api_key_here" ]; then
    echo "❌ GEMINI_API_KEY not set in .env file"
    echo "   Please edit .env and add your GEMINI_API_KEY"
    exit 1
fi

echo "✓ .env file found with API key"
echo ""

# Start backend
echo "Starting backend server..."
cd backend
python3 main.py > ../backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 5

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓ Backend is running on http://localhost:8000"
else
    echo "⚠️  Backend may not have started properly"
    echo "   Check backend.log for errors"
fi

echo ""
echo "Starting frontend..."
npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!

echo ""
echo "=========================================="
echo "Servers starting..."
echo "=========================================="
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Logs:"
echo "  Backend:  tail -f backend.log"
echo "  Frontend: tail -f frontend.log"
echo ""
echo "To stop servers:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo ""

