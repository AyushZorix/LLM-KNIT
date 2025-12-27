# KNIT-LLM

A dual-pipeline LLM system for reduction of hallucinations using HyDE (Hypothetical Document Embeddings) and semantic-heuristic processing.

## Features

- 🎨 Modern, clean UI with smooth animations
- 💬 Real-time message display with typing indicators
- 📱 Responsive design
- ⌨️ Keyboard shortcuts (Enter to send, Shift+Enter for new line)
- 🎯 Auto-scrolling to latest messages
- 🧹 Clear chat functionality
- 📝 Message timestamps
- 🤖 **Dual-Pipeline ML Architecture** for hallucination reduction

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
- Generates multiple candidate answers using Gemini

### Final Selection Stage
- Computes confidence scores for each candidate based on:
  - Semantic relevance to query
  - Consistency across generated answers
  - Similarity with HyDE embedding
- Selects and returns highest-confidence answer

## Project Structure

```
knit-llm/
├── src/                    # Frontend (React + TypeScript)
│   ├── components/         # React components
│   ├── services/          # API services
│   └── types/             # TypeScript types
├── backend/               # Backend (Python + FastAPI)
│   ├── pipeline1_hyde.py  # HyDE pipeline
│   ├── pipeline2_semantic.py  # Semantic processing
│   ├── final_selection.py # Selection logic
│   └── main.py            # FastAPI server
├── .env                   # Environment variables (create from .env.example)
└── package.json           # Frontend dependencies
```

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Python 3.9+
- Google Gemini API key

### Frontend Setup

1. **Install dependencies:**
```bash
npm install
```

2. **Start development server:**
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Backend Setup

1. **Install Python dependencies:**
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
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

4. **Start backend server:**
```bash
cd backend
python main.py
```

The backend will be available at `http://localhost:8000`

## Configuration

Create a `.env` file in the root directory:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional
GEMINI_MODEL=gemini-1.5-flash
BACKEND_PORT=8000
FRONTEND_PORT=3000
TOP_K_TOKENS=10
NUM_TRANSFORMER_CANDIDATES=5
```

## How It Works

1. **User sends a query** through the frontend
2. **Pipeline 1** generates a hypothetical answer using Gemini and creates embeddings
3. **Pipeline 2** processes the query semantically:
   - Tokenizes and selects important words
   - Builds MST to identify domain
   - Generates multiple candidate answers
4. **Final Selection** computes confidence scores and selects the best answer
5. **Response** is returned to the user

## Hallucination Reduction

The system reduces hallucinations through:

1. **HyDE Embeddings**: Provides a semantic anchor for answer quality
2. **Multiple Candidates**: Generates several answers to compare consistency
3. **Domain Classification**: Ensures answers are relevant to the query domain
4. **Confidence Scoring**: Selects answers with highest semantic relevance and consistency
5. **Cross-validation**: Compares candidates against HyDE and each other

## Development

### Frontend Development
```bash
npm run dev
```

### Backend Development
```bash
cd backend
python main.py
```

### Build for Production
```bash
# Frontend
npm run build

# Backend
# Deploy the backend/ directory to your server
```

## License

MIT
