"""
Local LLM integration using Ollama for response generation.
"""

import logging
import ollama
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class LLMGenerator:
    """
    Generates responses using local Ollama LLM.
    """
    
    def __init__(self, model_name: str = "llama3.2:3b"):
        """
        Initialize the LLM generator.
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama."""
        try:
            ollama.list()
            logger.info(f"Ollama connection successful, using model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise
    
    def generate_response(self, query: str, context_chunks: List[str], 
                         max_tokens: int = 500) -> str:
        """
        Generate response using retrieved context.
        
        Args:
            query: User query
            context_chunks: Retrieved document chunks
            max_tokens: Maximum response length
            
        Returns:
            Generated response
        """
        try:
            # Build prompt with context
            context = "\n\n".join(context_chunks[:3]) if context_chunks else ""
            
            if context:
                prompt = f"""Based on the following context, answer the user's question accurately and concisely.

Context:
{context}

Question: {query}

Answer:"""
            else:
                prompt = f"""Answer the following question using your general knowledge:

Question: {query}

Answer:"""
            
            # Generate response
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'num_predict': max_tokens,
                    'temperature': 0.7,
                    'top_p': 0.9
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_complex_response(self, query: str, reasoning_steps: List[Dict[str, Any]]) -> str:
        """
        Generate response for complex reasoning queries.
        
        Args:
            query: Original complex query
            reasoning_steps: List of reasoning steps with results
            
        Returns:
            Synthesized response
        """
        try:
            # Build context from reasoning steps
            context_parts = []
            for step in reasoning_steps:
                if step.get('response'):
                    context_parts.append(f"Step {step['step_number']}: {step['response']}")
            
            context = "\n\n".join(context_parts)
            
            prompt = f"""Based on the following multi-step analysis, provide a comprehensive answer to the original question.

Analysis Steps:
{context}

Original Question: {query}

Synthesized Answer:"""
            
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    'num_predict': 600,
                    'temperature': 0.6
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"Failed to generate complex response: {e}")
            return f"Error generating complex response: {str(e)}"