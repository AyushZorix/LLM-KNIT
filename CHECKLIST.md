# KNIT-LLM Setup Checklist

## ✅ Pre-Flight Checks

Run the verification script to check everything:
```bash
cd backend
python verify_setup.py
```

## Required Setup Steps

### 1. Environment File ✓
- [ ] Create `.env` file in project root (copy from `.env.example`)
- [ ] Add your `GEMINI_API_KEY` to `.env`
- [ ] Verify file is in root directory: `/Users/ayushbhandari/Downloads/knit-llm/.env`

### 2. Backend Dependencies ✓
- [ ] Install Python packages: `pip install -r requirements.txt`
- [ ] Download spaCy model: `python -m spacy download en_core_web_md`
- [ ] Verify all imports work

### 3. Frontend Dependencies ✓
- [ ] Install npm packages: `npm install`
- [ ] Verify `VITE_API_URL` in `.env` points to backend (default: `http://localhost:8000`)

### 4. Start Services ✓
- [ ] Start backend: `cd backend && python main.py`
- [ ] Start frontend: `npm run dev`
- [ ] Verify backend is accessible at `http://localhost:8000/health`
- [ ] Verify frontend is accessible at `http://localhost:3000`

## File Structure Verification

```
knit-llm/
├── .env                    ← MUST EXIST with GEMINI_API_KEY
├── .env.example
├── backend/
│   ├── main.py            ← FastAPI server
│   ├── pipeline1_hyde.py  ← HyDE pipeline
│   ├── pipeline2_semantic.py ← Semantic processing
│   ├── final_selection.py ← Selection logic
│   ├── requirements.txt
│   └── verify_setup.py    ← Verification script
└── src/
    ├── services/
    │   └── api.ts         ← Frontend API client
    └── components/
        └── ChatContainer.tsx ← Uses real API
```

## Common Issues & Fixes

### Issue: "GEMINI_API_KEY not found"
**Fix:** 
- Ensure `.env` file exists in project root
- Check that `GEMINI_API_KEY=your_key_here` is in `.env`
- No quotes around the key value

### Issue: "Module not found"
**Fix:**
```bash
cd backend
pip install -r requirements.txt
```

### Issue: "spaCy model not found"
**Fix:**
```bash
python -m spacy download en_core_web_md
```

### Issue: "Cannot connect to API" (Frontend)
**Fix:**
- Verify backend is running: `curl http://localhost:8000/health`
- Check `VITE_API_URL` in `.env` matches backend URL
- Check CORS settings in `backend/main.py`

### Issue: "CORS error"
**Fix:**
- Backend already configured for `localhost:3000` and `localhost:5173`
- If using different port, add it to `allow_origins` in `backend/main.py`

## Testing

### Test Backend Health
```bash
curl http://localhost:8000/health
```

### Test API Endpoint
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the capital of France?"}'
```

### Test Frontend
1. Open `http://localhost:3000`
2. Type a message
3. Check browser console for errors
4. Check backend terminal for processing logs

## Expected Behavior

When you send a query:
1. Backend receives request
2. Pipeline 1 generates HyDE response
3. Pipeline 2 processes semantically and generates candidates
4. Final selection picks best answer
5. Response returned to frontend
6. Message appears in chat

You should see logs in backend terminal showing:
- "🔄 Pipeline 1: Generating HyDE response..."
- "🔄 Pipeline 2: Semantic + Heuristic Processing..."
- "🔄 Final Selection: Computing confidence scores..."
- "✓ Response generated successfully"

## Next Steps After Setup

1. ✅ Run verification: `python backend/verify_setup.py`
2. ✅ Start backend: `cd backend && python main.py`
3. ✅ Start frontend: `npm run dev`
4. ✅ Test with a query in the UI
5. ✅ Check backend logs for processing details

