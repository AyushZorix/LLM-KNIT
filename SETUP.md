# KNIT-LLM Setup Guide

## Quick Start

### 1. Create Environment File

```bash
cp .env.example .env
```

Edit `.env` and add your Gemini API key:
```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 2. Setup Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### 3. Start Backend Server

```bash
# From backend directory
python main.py

# Or use the start script
./start.sh
```

Backend will run on `http://localhost:8000`

### 4. Setup Frontend

```bash
# From root directory
npm install
npm run dev
```

Frontend will run on `http://localhost:3000`

## How It Works

### System Flow

1. **User Query** → Frontend sends query to backend
2. **Pipeline 1 (HyDE)**:
   - Gemini generates hypothetical answer
   - Creates embeddings for the answer
3. **Pipeline 2 (Semantic + Heuristic)**:
   - Tokenizes query
   - Selects top-k important words
   - Builds Maximum Spanning Tree (MST)
   - Classifies domain
   - Generates 5 candidate answers using Gemini
4. **Final Selection**:
   - Computes confidence scores for each candidate
   - Scores based on:
     - Semantic relevance to query (35%)
     - Consistency across answers (25%)
     - Similarity to HyDE embedding (20%)
     - Base confidence (10%)
     - Perplexity score (10%)
   - Returns highest-confidence answer

### Hallucination Reduction

The system reduces hallucinations through:

1. **HyDE Embeddings**: Provides semantic anchor for answer quality
2. **Multiple Candidates**: Generates 5 answers to check consistency
3. **Domain Classification**: Ensures domain relevance
4. **Confidence Scoring**: Multi-factor scoring system
5. **Cross-validation**: Compares candidates against HyDE and each other

## Troubleshooting

### Backend Issues

**Error: GEMINI_API_KEY not found**
- Make sure `.env` file exists in root directory
- Check that `GEMINI_API_KEY` is set correctly

**Error: spaCy model not found**
```bash
python -m spacy download en_core_web_md
```

**Error: Module not found**
```bash
pip install -r requirements.txt
```

### Frontend Issues

**Error: Cannot connect to API**
- Make sure backend is running on port 8000
- Check `VITE_API_URL` in `.env` matches backend URL
- Check CORS settings in `backend/main.py`

**Error: Network error**
- Verify backend is accessible at `http://localhost:8000/health`
- Check browser console for detailed error messages

## API Endpoints

### `POST /api/chat`
Main chat endpoint.

**Request:**
```json
{
  "message": "What is the capital of France?",
  "history": []
}
```

**Response:**
```json
{
  "message": "The capital of France is Paris...",
  "timestamp": "2024-01-01T12:00:00",
  "metadata": {
    "selected_id": 1,
    "confidence": 0.92,
    "domain": "geography",
    "num_candidates": 5
  }
}
```

## Configuration Options

Edit `.env` to customize:

- `GEMINI_MODEL`: Model to use (default: `gemini-1.5-flash`)
- `TOP_K_TOKENS`: Number of top tokens (default: `10`)
- `NUM_TRANSFORMER_CANDIDATES`: Number of candidates (default: `5`)
- `BACKEND_PORT`: Backend port (default: `8000`)
- `FRONTEND_PORT`: Frontend port (default: `3000`)

## Testing

### Test Backend
```bash
curl http://localhost:8000/health
```

### Test API
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the capital of France?"}'
```

## Production Deployment

1. **Backend**: Deploy `backend/` directory to your server
2. **Frontend**: Build and deploy:
   ```bash
   npm run build
   # Deploy dist/ directory
   ```
3. **Environment**: Set production environment variables
4. **CORS**: Update CORS origins in `backend/main.py`

