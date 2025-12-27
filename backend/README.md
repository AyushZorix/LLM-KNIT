# KNIT-LLM Backend

Backend API server implementing the dual-pipeline architecture for hallucination reduction in LLMs.

## Architecture

### Pipeline 1: HyDE-based Generation
- Takes user query as input
- Calls Gemini API to generate a hypothetical answer
- Generates and stores HyDE embeddings

### Pipeline 2: Semantic + Heuristic Processing
- Tokenizes the user query
- Selects top-k important words
- Assigns tokens to domains/categories
- Builds Maximum Spanning Tree (MST) over tokens
- Classifies dominant domain
- Generates multiple candidate answers using Transformer (Gemini)

### Final Selection Stage
- Computes confidence scores for each candidate based on:
  - Semantic relevance to query
  - Consistency across generated answers
  - Similarity with HyDE embedding
- Selects and returns highest-confidence answer

## Setup

1. **Install dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

2. **Download spaCy model:**
```bash
python -m spacy download en_core_web_md
```

3. **Create `.env` file:**
```bash
cp ../.env.example ../.env
# Edit .env and add your GEMINI_API_KEY
```

4. **Run the server:**
```bash
python main.py
```

The server will start on `http://localhost:8000` by default.

## API Endpoints

### `GET /`
Root endpoint with API information.

### `GET /health`
Health check endpoint.

### `POST /api/chat`
Main chat endpoint.

**Request:**
```json
{
  "message": "What is the capital of France?",
  "history": [
    {
      "role": "user",
      "content": "Hello"
    },
    {
      "role": "assistant",
      "content": "Hi! How can I help you?"
    }
  ]
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

## Environment Variables

See `.env.example` for all available configuration options.

Required:
- `GEMINI_API_KEY`: Your Google Gemini API key

Optional:
- `GEMINI_MODEL`: Model name (default: `gemini-1.5-flash`)
- `BACKEND_PORT`: Server port (default: `8000`)
- `TOP_K_TOKENS`: Number of top tokens to select (default: `10`)
- `NUM_TRANSFORMER_CANDIDATES`: Number of candidate answers (default: `5`)

