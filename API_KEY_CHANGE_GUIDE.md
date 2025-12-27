# API Key Change Commands

## When you change your Gemini API key, follow these steps:

### Step 1: Update the .env file
```bash
# Edit the .env file with your new API key
nano .env
# or
code .env
```

Update this line:
```
GEMINI_API_KEY="your-new-api-key-here"
```

### Step 2: Restart the Backend (REQUIRED)
```bash
# Kill any existing backend processes
pkill -f "main.py" || true

# Navigate to backend directory and restart
cd backend
../.venv/bin/python main.py
```

### Step 3: Verify Backend is Running
```bash
# Test if backend is responding
curl -s http://localhost:8000/health
```
You should see: `{"status":"healthy"}`

### Step 4: Test with Frontend
- Frontend doesn't need restarting
- Just try making a query in your browser at http://localhost:3000

## Alternative One-Line Commands:

### Quick Backend Restart:
```bash
pkill -f "main.py" && cd backend && ../.venv/bin/python main.py
```

### Full System Restart (if needed):
```bash
# Stop everything
pkill -f "main.py" && pkill -f "npm"

# Start backend
cd backend && ../.venv/bin/python main.py &

# Start frontend (in new terminal)
npm run dev
```

## Why This is Necessary:
- Environment variables (.env) are only loaded when the Python process starts
- The backend server caches the API key in memory
- Changing .env while server is running has no effect
- Only restarting the backend picks up the new API key