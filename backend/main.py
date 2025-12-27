"""
Main FastAPI server for KNIT-LLM backend.
Integrates Pipeline 1 (HyDE), Pipeline 2 (Semantic+Heuristic), and Final Selection.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import os
import traceback
import json
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# Load .env from parent directory (project root)
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)
# Also try loading from current directory as fallback
load_dotenv()

from pipeline1_hyde import HyDEPipeline
from pipeline2_semantic import SemanticHeuristicPipeline
from final_selection import FinalSelector

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, (np.integer, np.int_, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif type(obj).__module__ == 'numpy':
            try:
                return float(obj) if 'float' in str(type(obj)) else int(obj)
            except:
                return str(obj)
        return super().default(obj)

app = FastAPI(
    title="KNIT-LLM API", 
    version="1.0.0"
)

# CORS middleware - Allow all origins in development (more permissive for debugging)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now
    allow_credentials=False,  # Set to False when using allow_origins=["*"]
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Add explicit OPTIONS handler for preflight requests
@app.options("/api/chat")
async def preflight_handler():
    """Handle preflight OPTIONS requests."""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Initialize pipelines
try:
    hyde_pipeline = HyDEPipeline()
    semantic_pipeline = SemanticHeuristicPipeline()
    final_selector = FinalSelector()
    print("✓ All pipelines initialized successfully")
except Exception as e:
    print(f"✗ Error initializing pipelines: {e}")
    traceback.print_exc()
    raise

class QueryRequest(BaseModel):
    """Request model for chat query."""
    message: str
    history: Optional[List[Dict[str, str]]] = None

class QueryResponse(BaseModel):
    """Response model for chat query."""
    message: str
    timestamp: str
    metadata: Optional[Dict] = None

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "KNIT-LLM API",
        "version": "1.0.0",
        "status": "running",
        "pipelines": {
            "pipeline1_hyde": "active",
            "pipeline2_semantic": "active",
            "final_selection": "active"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/api/chat")
async def chat(request: QueryRequest):
    """
    Main chat endpoint that processes queries through the dual-pipeline architecture.
    
    Flow:
    1. Pipeline 1: Generate HyDE response and embeddings
    2. Pipeline 2: Semantic + Heuristic processing, generate candidate answers
    3. Final Selection: Select best answer based on confidence scores
    """
    try:
        query = request.message.strip()
        
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        print(f"\n{'='*70}")
        print(f"Processing query: {query}")
        print(f"{'='*70}\n")
        
        # Pipeline 1: HyDE Generation
        hyde_result = hyde_pipeline.process(query)
        hyde_embedding = np.array(hyde_result["hyde_embedding"])
        
        # Pipeline 2: Semantic + Heuristic Processing
        semantic_result = semantic_pipeline.process(query)
        candidates = semantic_result["candidates"]
        
        # Add domain relevance to candidates
        domain = semantic_result["domain"]
        for candidate in candidates:
            candidate["domain_relevance"] = {
                domain: 0.9 if candidate.get("domain") == domain else 0.7
            }
            # Add task metrics (simplified)
            candidate["metrics"] = {
                "fluency": 0.8,
                "coherence": 0.8,
                "completeness": 0.75
            }
        
        # Final Selection: Select best answer
        best_candidate, selection_metadata = final_selector.select_best(
            query,
            hyde_embedding,
            candidates
        )
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, (np.integer, np.int_, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(key): convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            elif type(obj).__module__ == 'numpy':
                try:
                    return float(obj) if isinstance(obj, (np.floating, np.float32, np.float64)) else int(obj)
                except:
                    return str(obj)
            return obj
        
        # Convert all metadata to ensure no numpy types
        clean_metadata = convert_numpy_types(selection_metadata)
        
        # Prepare response
        response_dict = {
            "message": str(best_candidate["text"]),
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "selected_id": int(clean_metadata.get("selected_id", 0)),
                "confidence": float(clean_metadata.get("confidence", 0.0)),
                "domain": str(domain),
                "hyde_response": str(hyde_result["hyde_response"][:200] + "..." if len(hyde_result["hyde_response"]) > 200 else hyde_result["hyde_response"]),
                "num_candidates": int(len(candidates)),
                "selection_metadata": convert_numpy_types(clean_metadata)
            }
        }
        
        print(f"\n{'='*70}")
        print(f"✓ Response generated successfully")
        print(f"  Confidence: {float(selection_metadata.get('confidence', 0)):.3f}")
        print(f"  Domain: {domain}")
        print(f"{'='*70}\n")
        
        # Convert all numpy types
        clean_response = convert_numpy_types(response_dict)
        
        # Return as JSONResponse with explicit CORS headers
        return JSONResponse(
            content=clean_response,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )
        
    except Exception as e:
        print(f"Error processing query: {e}")
        traceback.print_exc()
        # Return error with CORS headers
        return JSONResponse(
            status_code=500,
            content={"detail": f"Error processing query: {str(e)}"},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )

if __name__ == "__main__":
    port = int(os.getenv("BACKEND_PORT", "8000"))
    host = os.getenv("BACKEND_HOST", "0.0.0.0")  # Bind to all interfaces
    
    print(f"\n{'='*70}")
    print(f"Starting KNIT-LLM Backend Server")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"{'='*70}\n")
    
    uvicorn.run(app, host=host, port=port)

