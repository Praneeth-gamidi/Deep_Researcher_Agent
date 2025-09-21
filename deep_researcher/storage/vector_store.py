"""
Vector storage and similarity search using FAISS.
"""

import numpy as np
import faiss
import pickle
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector storage and similarity search using FAISS.
    """
    
    def __init__(self, dimension: int, index_type: str = "flat", persist_dir: str = "data/embeddings"):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the embedding vectors
            index_type: Type of FAISS index ("flat", "ivf", "hnsw")
            persist_dir: Directory to persist the index
        """
        self.dimension = dimension
        self.index_type = index_type
        self.persist_dir = persist_dir
        self.index = None
        self.metadata = []
        self.id_to_metadata = {}
        self.next_id = 0
        
        # Create persist directory
        os.makedirs(persist_dir, exist_ok=True)
        
        # Initialize FAISS index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize the FAISS index based on the specified type."""
        try:
            if self.index_type == "flat":
                # Flat index for exact search - use L2 distance for better compatibility
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == "ivf":
                # IVF index for approximate search
                quantizer = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
            elif self.index_type == "hnsw":
                # HNSW index for approximate search
                self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            logger.info(f"Initialized {self.index_type} index with dimension {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to initialize index: {e}")
            raise
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]) -> List[int]:
        """
        Add vectors to the index.
        
        Args:
            vectors: Array of vectors to add
            metadata: List of metadata dictionaries for each vector
            
        Returns:
            List of assigned IDs
        """
        try:
            if vectors.shape[1] != self.dimension:
                raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dimension}")
            
            # Normalize vectors for L2 distance
            faiss.normalize_L2(vectors)
            
            # Get IDs for the vectors
            ids = list(range(self.next_id, self.next_id + len(vectors)))
            self.next_id += len(vectors)
            
            # Add to index
            if self.index_type == "ivf" and not self.index.is_trained:
                # Train IVF index if not already trained
                self.index.train(vectors)
            
            # Use simple add method for better compatibility
            self.index.add(vectors)
            
            # Store metadata
            for i, meta in enumerate(metadata):
                meta['id'] = ids[i]
                self.metadata.append(meta)
                self.id_to_metadata[ids[i]] = meta
            
            logger.info(f"Added {len(vectors)} vectors to index")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            raise
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            
        Returns:
            List of (id, score, metadata) tuples
        """
        try:
            if query_vector.shape[0] != self.dimension:
                raise ValueError(f"Query dimension {query_vector.shape[0]} doesn't match index dimension {self.dimension}")
            
            # Normalize query vector
            query_vector = query_vector.reshape(1, -1)
            faiss.normalize_L2(query_vector)
            
            # Search
            scores, indices = self.index.search(query_vector, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and idx < len(self.metadata):  # Valid result
                    # Convert L2 distance to similarity score (lower distance = higher similarity)
                    similarity_score = 1.0 / (1.0 + score) if score > 0 else 1.0
                    metadata = self.metadata[idx] if idx < len(self.metadata) else {}
                    results.append((int(idx), float(similarity_score), metadata))
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []
    
    def get_vector(self, vector_id: int) -> Optional[np.ndarray]:
        """
        Get a vector by its ID.
        
        Args:
            vector_id: ID of the vector to retrieve
            
        Returns:
            Vector array or None if not found
        """
        try:
            if vector_id not in self.id_to_metadata:
                return None
            
            # FAISS doesn't support direct retrieval by ID, so we need to search
            # This is a limitation of FAISS - in production, you might want to use a different approach
            logger.warning("Direct vector retrieval by ID not supported by FAISS")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get vector {vector_id}: {e}")
            return None
    
    def get_metadata(self, vector_id: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a vector by its ID.
        
        Args:
            vector_id: ID of the vector
            
        Returns:
            Metadata dictionary or None if not found
        """
        return self.id_to_metadata.get(vector_id)
    
    def remove_vector(self, vector_id: int) -> bool:
        """
        Remove a vector from the index.
        
        Args:
            vector_id: ID of the vector to remove
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if vector_id not in self.id_to_metadata:
                return False
            
            # FAISS doesn't support direct removal, so we need to rebuild the index
            # This is a limitation of FAISS
            logger.warning("Vector removal not directly supported by FAISS - would require index rebuild")
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove vector {vector_id}: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dictionary containing index statistics
        """
        try:
            stats = {
                'total_vectors': self.index.ntotal,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'is_trained': getattr(self.index, 'is_trained', True)
            }
            
            if hasattr(self.index, 'nlist'):
                stats['nlist'] = self.index.nlist
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {}
    
    def save_index(self, filename: str = "vector_index"):
        """
        Save the index to disk.
        
        Args:
            filename: Base filename for the index files
        """
        try:
            index_path = os.path.join(self.persist_dir, f"{filename}.faiss")
            metadata_path = os.path.join(self.persist_dir, f"{filename}_metadata.pkl")
            
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'id_to_metadata': self.id_to_metadata,
                    'next_id': self.next_id
                }, f)
            
            logger.info(f"Index saved to {index_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def load_index(self, filename: str = "vector_index") -> bool:
        """
        Load the index from disk.
        
        Args:
            filename: Base filename for the index files
            
        Returns:
            True if successful, False otherwise
        """
        try:
            index_path = os.path.join(self.persist_dir, f"{filename}.faiss")
            metadata_path = os.path.join(self.persist_dir, f"{filename}_metadata.pkl")
            
            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                logger.warning(f"Index files not found: {filename}")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.id_to_metadata = data['id_to_metadata']
                self.next_id = data['next_id']
            
            logger.info(f"Index loaded from {index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def clear(self):
        """Clear all vectors from the index."""
        try:
            self._initialize_index()
            self.metadata = []
            self.id_to_metadata = {}
            self.next_id = 0
            logger.info("Index cleared")
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")
