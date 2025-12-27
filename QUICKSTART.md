# Quick Start Guide

## The Error You're Seeing

If you see:
```
Could not connect to the server
Fetch API cannot load http://localhost:8000/api/chat due to access control checks
```

**This means the backend is not running!**

## Solution: Start the Backend

### Step 1: Check if .env file exists
```bash
# From project root
ls -la .env
```

If it doesn't exist:
```bash
cp .env.example .env
# Then edit .env and add your GEMINI_API_KEY
```

### Step 2: Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### Step 3: Start the Backend Server
```bash
# From backend directory
python main.py
```

You should see:
```
======================================================================
Starting KNIT-LLM Backend Server
Host: localhost
Port: 8000
======================================================================

✓ All pipelines initialized successfully
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:8000
```

### Step 4: Keep Backend Running
**Keep this terminal open!** The backend must stay running.

### Step 5: Start Frontend (in a NEW terminal)
```bash
# From project root
npm run dev
```

### Step 6: Test
1. Open browser to `http://localhost:3000` (or the port shown)
2. Send a message
3. Check backend terminal for processing logs

## Verify Backend is Running

Test in a new terminal:
```bash
curl http://localhost:8000/health
```

Should return: `{"status":"healthy"}`

## Common Issues

### "GEMINI_API_KEY not found"
- Make sure `.env` file exists in project root
- Check that it contains: `GEMINI_API_KEY=your_key_here`
- No quotes around the key

### "Module not found"
```bash
cd backend
pip install -r requirements.txt
```

### "spaCy model not found"
```bash
python -m spacy download en_core_web_md
```

### Backend starts but frontend still can't connect
1. Check backend is actually running: `curl http://localhost:8000/health`
2. Check what port frontend is using (should be 3000 or 5173)
3. Make sure you're using the correct URL in frontend

## Two Terminal Windows Needed

**Terminal 1 (Backend):**
```bash
cd backend
python main.py
# Keep this running!
```

**Terminal 2 (Frontend):**
```bash
npm run dev
# Keep this running!
```

Both must be running simultaneously!

