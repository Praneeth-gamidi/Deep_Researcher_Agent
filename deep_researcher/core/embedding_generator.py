"""
Local embedding generation using sentence-transformers.
"""

import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional, Union
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings for text using local sentence-transformers models.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "data/embeddings"):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model = None
        self.embedding_cache = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        if not text.strip():
            return np.zeros(self.model.get_sentence_embedding_dimension())
        
        try:
            # Check cache first
            text_hash = hash(text)
            if text_hash in self.embedding_cache:
                return self.embedding_cache[text_hash]
            
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Cache the result
            self.embedding_cache[text_hash] = embedding
            
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return np.zeros(self.model.get_sentence_embedding_dimension())
    
    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Filter out empty texts
            non_empty_texts = [text for text in texts if text.strip()]
            if not non_empty_texts:
                return [np.zeros(self.model.get_sentence_embedding_dimension()) for _ in texts]
            
            # Generate embeddings in batches
            embeddings = self.model.encode(
                non_empty_texts, 
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            
            # Create result list with zeros for empty texts
            result = []
            text_idx = 0
            for text in texts:
                if text.strip():
                    result.append(embeddings[text_idx])
                    text_idx += 1
                else:
                    result.append(np.zeros(self.model.get_sentence_embedding_dimension()))
            
            return result
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return [np.zeros(self.model.get_sentence_embedding_dimension()) for _ in texts]
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         candidate_embeddings: List[np.ndarray], 
                         top_k: int = 5) -> List[tuple]:
        """
        Find the most similar embeddings to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embedding vectors
            top_k: Number of top similar embeddings to return
            
        Returns:
            List of (index, similarity_score) tuples sorted by similarity
        """
        if not candidate_embeddings:
            return []
        
        try:
            similarities = []
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.compute_similarity(query_embedding, candidate)
                similarities.append((i, similarity))
            
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
        except Exception as e:
            logger.error(f"Failed to find similar embeddings: {e}")
            return []
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], filename: str):
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Dictionary mapping text IDs to embeddings
            filename: Filename to save to
        """
        try:
            filepath = os.path.join(self.cache_dir, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"Embeddings saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
    
    def load_embeddings(self, filename: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings from disk.
        
        Args:
            filename: Filename to load from
            
        Returns:
            Dictionary mapping text IDs to embeddings
        """
        try:
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    embeddings = pickle.load(f)
                logger.info(f"Embeddings loaded from {filepath}")
                return embeddings
            else:
                logger.warning(f"Embeddings file not found: {filepath}")
                return {}
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return {}
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")
