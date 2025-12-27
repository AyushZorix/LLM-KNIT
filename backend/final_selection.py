"""
Final Selection Stage
Computes confidence scores and selects the best answer.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load .env from parent directory (project root)
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)
# Also try loading from current directory as fallback
load_dotenv()

class FinalSelector:
    """Final selection stage that combines HyDE and candidate answers."""
    
    def __init__(self):
        embedding_model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(embedding_model_name)
    
    def compute_semantic_relevance(self, answer: str, query: str) -> float:
        """
        Compute semantic relevance between answer and query.
        
        Args:
            answer: Candidate answer text
            query: Original query
            
        Returns:
            Relevance score (0-1)
        """
        try:
            query_embedding = self.embedding_model.encode(query, normalize_embeddings=True)
            answer_embedding = self.embedding_model.encode(answer, normalize_embeddings=True)
            
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                answer_embedding.reshape(1, -1)
            )[0][0]
            
            # Convert numpy type to Python float and normalize to 0-1 range
            similarity_float = float(similarity)
            return float(max(0.0, min(1.0, (similarity_float + 1) / 2)))
        except Exception as e:
            print(f"Error computing semantic relevance: {e}")
            return 0.5
    
    def compute_consistency(self, candidates: List[Dict[str, any]]) -> Dict[int, float]:
        """
        Compute consistency scores across candidates.
        
        Args:
            candidates: List of candidate answer dictionaries
            
        Returns:
            Dictionary mapping candidate ID to consistency score
        """
        if len(candidates) < 2:
            return {c["id"]: 1.0 for c in candidates}
        
        # Generate embeddings for all candidates
        texts = [c["text"] for c in candidates]
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)
        
        consistency_scores = {}
        
        for i, candidate in enumerate(candidates):
            # Compute average similarity with all other candidates
            similarities = []
            for j, other_candidate in enumerate(candidates):
                if i != j:
                    sim = cosine_similarity(
                        embeddings[i].reshape(1, -1),
                        embeddings[j].reshape(1, -1)
                    )[0][0]
                    similarities.append(float(sim))  # Convert numpy to Python float
            
            # Consistency = average similarity (higher = more consistent)
            consistency = np.mean(similarities) if similarities else 0.5
            consistency_scores[candidate["id"]] = float(max(0.0, min(1.0, (consistency + 1) / 2)))
        
        return consistency_scores
    
    def compute_hyde_similarity(self, answer: str, hyde_embedding: np.ndarray) -> float:
        """
        Compute similarity between answer and HyDE embedding.
        
        Args:
            answer: Candidate answer text
            hyde_embedding: HyDE embedding vector
            
        Returns:
            Similarity score (0-1)
        """
        try:
            answer_embedding = self.embedding_model.encode(answer, normalize_embeddings=True)
            
            similarity = cosine_similarity(
                hyde_embedding.reshape(1, -1),
                answer_embedding.reshape(1, -1)
            )[0][0]
            
            # Convert numpy type to Python float and normalize to 0-1 range
            similarity_float = float(similarity)
            return float(max(0.0, min(1.0, (similarity_float + 1) / 2)))
        except Exception as e:
            print(f"Error computing HyDE similarity: {e}")
            return 0.5
    
    def compute_confidence_score(
        self,
        candidate: Dict[str, any],
        query: str,
        hyde_embedding: np.ndarray,
        consistency_score: float,
        semantic_relevance: float,
        hyde_similarity: float
    ) -> float:
        """
        Compute final confidence score for a candidate.
        
        Args:
            candidate: Candidate answer dictionary
            query: Original query
            hyde_embedding: HyDE embedding
            consistency_score: Consistency with other candidates
            semantic_relevance: Relevance to query
            hyde_similarity: Similarity to HyDE
            
        Returns:
            Final confidence score (0-1)
        """
        # Base confidence from candidate
        base_confidence = candidate.get("confidence", 0.7)
        
        # Perplexity score (lower is better)
        perplexity = candidate.get("perplexity", 15.0)
        perplexity_score = 1.0 / (1.0 + perplexity / 50.0)
        
        # Weighted combination
        weights = {
            "semantic_relevance": 0.35,
            "consistency": 0.25,
            "hyde_similarity": 0.20,
            "base_confidence": 0.10,
            "perplexity_score": 0.10
        }
        
        confidence = (
            semantic_relevance * weights["semantic_relevance"] +
            consistency_score * weights["consistency"] +
            hyde_similarity * weights["hyde_similarity"] +
            base_confidence * weights["base_confidence"] +
            perplexity_score * weights["perplexity_score"]
        )
        
        return float(max(0.0, min(1.0, confidence)))
    
    def select_best(
        self,
        query: str,
        hyde_embedding: np.ndarray,
        candidates: List[Dict[str, any]]
    ) -> Tuple[Dict[str, any], Dict[str, any]]:
        """
        Select the best candidate answer.
        
        Args:
            query: Original query
            hyde_embedding: HyDE embedding vector
            candidates: List of candidate answers
            
        Returns:
            Tuple of (best_candidate, selection_metadata)
        """
        print("🔄 Final Selection: Computing confidence scores...")
        
        if not candidates:
            raise ValueError("No candidates provided")
        
        if len(candidates) == 1:
            return candidates[0], {"reason": "Only one candidate available"}
        
        # Compute consistency scores
        consistency_scores = self.compute_consistency(candidates)
        print(f"✓ Computed consistency scores")
        
        # Compute scores for each candidate
        candidate_scores = []
        
        for candidate in candidates:
            # Semantic relevance
            semantic_relevance = self.compute_semantic_relevance(candidate["text"], query)
            
            # Consistency
            consistency = consistency_scores.get(candidate["id"], 0.5)
            
            # HyDE similarity
            hyde_similarity = self.compute_hyde_similarity(candidate["text"], hyde_embedding)
            
            # Final confidence
            confidence = self.compute_confidence_score(
                candidate,
                query,
                hyde_embedding,
                consistency,
                semantic_relevance,
                hyde_similarity
            )
            
            candidate_scores.append({
                "candidate": candidate,
                "confidence": float(confidence),
                "semantic_relevance": float(semantic_relevance),
                "consistency": float(consistency),
                "hyde_similarity": float(hyde_similarity)
            })
        
        # Sort by confidence
        candidate_scores.sort(key=lambda x: x["confidence"], reverse=True)
        
        best = candidate_scores[0]
        best_candidate = best["candidate"]
        
        print(f"✓ Selected candidate #{best_candidate['id']} with confidence {best['confidence']:.3f}")
        
        # Prepare metadata - convert numpy types to native Python types
        def to_native_type(val):
            if isinstance(val, np.generic):
                return val.item()
            elif isinstance(val, (np.integer, np.int_, np.int32, np.int64)):
                return int(val)
            elif isinstance(val, (np.floating, np.float32, np.float64)):
                return float(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            elif hasattr(val, 'item'):
                return val.item()
            elif type(val).__module__ == 'numpy':
                try:
                    return float(val) if 'float' in str(type(val)) else int(val)
                except:
                    return float(val) if isinstance(val, (float, np.floating)) else int(val) if isinstance(val, (int, np.integer)) else str(val)
            return val
        
        metadata = {
            "selected_id": int(best_candidate["id"]),
            "confidence": float(to_native_type(best["confidence"])),
            "semantic_relevance": float(to_native_type(best["semantic_relevance"])),
            "consistency": float(to_native_type(best["consistency"])),
            "hyde_similarity": float(to_native_type(best["hyde_similarity"])),
            "all_scores": {
                int(cs["candidate"]["id"]): float(to_native_type(cs["confidence"]))
                for cs in candidate_scores
            },
            "runner_up_id": int(candidate_scores[1]["candidate"]["id"]) if len(candidate_scores) > 1 else None,
            "margin": float(to_native_type(best["confidence"] - candidate_scores[1]["confidence"] if len(candidate_scores) > 1 else 0.0))
        }
        
        # Final pass: recursively convert any remaining numpy types in metadata
        def final_convert(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: final_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [final_convert(item) for item in obj]
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        
        metadata = final_convert(metadata)
        
        return best_candidate, metadata

