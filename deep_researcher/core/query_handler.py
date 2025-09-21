"""
Query handling and response generation system.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ..storage.vector_store import VectorStore
from ..storage.document_store import DocumentStore
from ..core.embedding_generator import EmbeddingGenerator
from ..core.llm_generator import LLMGenerator

logger = logging.getLogger(__name__)


class QueryHandler:
    """
    Handles query processing and response generation.
    """
    
    def __init__(self, embedding_generator: EmbeddingGenerator, 
                 vector_store: VectorStore, document_store: DocumentStore,
                 llm_generator: Optional[LLMGenerator] = None):
        """
        Initialize the query handler.
        
        Args:
            embedding_generator: Embedding generator instance
            vector_store: Vector store instance
            document_store: Document store instance
            llm_generator: Optional LLM generator for better responses
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.document_store = document_store
        self.llm_generator = llm_generator
    
    def process_query(self, query: str, top_k: int = 5, 
                     include_metadata: bool = True) -> Dict[str, Any]:
        """
        Process a query and return relevant results.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            include_metadata: Whether to include metadata in results
            
        Returns:
            Dictionary containing query results and metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Search for similar vectors
            search_results = self.vector_store.search(query_embedding, k=top_k)
            
            # Process results
            results = []
            for vector_id, score, metadata in search_results:
                result = {
                    'vector_id': vector_id,
                    'similarity_score': score,
                    'chunk_text': metadata.get('chunk_text', ''),
                    'document_id': metadata.get('document_id'),
                    'chunk_index': metadata.get('chunk_index'),
                    'file_name': metadata.get('file_name', ''),
                    'file_path': metadata.get('file_path', '')
                }
                
                if include_metadata:
                    result['metadata'] = metadata
                
                results.append(result)
            
            # Generate response
            response = self._generate_response(query, results)
            
            # Store query in history
            self.document_store.add_query(
                query_text=query,
                response_text=response,
                metadata={'top_k': top_k, 'results_count': len(results)}
            )
            
            return {
                'query': query,
                'response': response,
                'results': results,
                'total_results': len(results),
                'top_k': top_k
            }
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            return {
                'query': query,
                'response': f"Error processing query: {str(e)}",
                'results': [],
                'total_results': 0,
                'error': str(e)
            }
    
    def _generate_response(self, query: str, results: List[Dict[str, Any]]) -> str:
        """
        Generate a response based on query and search results.
        
        Args:
            query: Original query
            results: Search results
            
        Returns:
            Generated response text
        """
        if not results:
            # Use LLM for general knowledge or fallback to template
            if self.llm_generator:
                return self.llm_generator.generate_response(query, [])
            else:
                return self._generate_empty_knowledge_base_response(query)
        
        # Extract relevant text chunks
        chunks = [result['chunk_text'] for result in results if result['chunk_text']]
        
        if not chunks:
            return "No relevant text content found for your query."
        
        # Use LLM if available, otherwise fallback to template
        if self.llm_generator:
            return self.llm_generator.generate_response(query, chunks)
        else:
            # Fallback to original template-based response
            response_parts = []
            response_parts.append(f"Based on your query '{query}', I found the following relevant information:")
            response_parts.append("")
            
            for i, chunk in enumerate(chunks[:3], 1):
                response_parts.append(f"{i}. {chunk[:500]}{'...' if len(chunk) > 500 else ''}")
                response_parts.append("")
            
            if len(results) > 0:
                sources = set()
                for result in results:
                    if result.get('file_name'):
                        sources.add(result['file_name'])
                
                if sources:
                    response_parts.append(f"Sources: {', '.join(sources)}")
            
            return "\n".join(response_parts)
    
    def _generate_empty_knowledge_base_response(self, query: str) -> str:
        """
        Generate a helpful response when the knowledge base is empty.
        
        Args:
            query: Original query
            
        Returns:
            Helpful response text
        """
        response_parts = []
        
        response_parts.append("I don't have any documents in my knowledge base yet, so I can't provide specific information about your query.")
        response_parts.append("")
        response_parts.append("However, I can help you in several ways:")
        response_parts.append("")
        response_parts.append("1. **Add Documents**: Use 'add <file_path>' to add documents to my knowledge base")
        response_parts.append("2. **Upload Files**: I can process PDF, TXT, MD, DOCX, and HTML files")
        response_parts.append("3. **General Knowledge**: I can still help with general questions using my built-in knowledge")
        response_parts.append("")
        response_parts.append("Would you like me to:")
        response_parts.append("• Help you add some documents?")
        response_parts.append("• Answer your question using general knowledge?")
        response_parts.append("• Show you how to get started?")
        response_parts.append("")
        response_parts.append("Type 'help' for more commands or ask me anything!")
        
        return "\n".join(response_parts)
    
    def get_similar_queries(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get similar queries from query history.
        
        Args:
            query: Query text
            limit: Maximum number of similar queries to return
            
        Returns:
            List of similar queries
        """
        try:
            # Get query history
            query_history = self.document_store.get_query_history(limit=100)
            
            if not query_history:
                return []
            
            # Generate embeddings for query and history
            query_embedding = self.embedding_generator.generate_embedding(query)
            history_queries = [q['query_text'] for q in query_history]
            history_embeddings = self.embedding_generator.generate_embeddings_batch(history_queries)
            
            # Find similar queries
            similar_queries = self.embedding_generator.find_most_similar(
                query_embedding, history_embeddings, top_k=limit
            )
            
            # Format results
            results = []
            for idx, score in similar_queries:
                if idx < len(query_history):
                    results.append({
                        'query': query_history[idx]['query_text'],
                        'similarity_score': score,
                        'created_at': query_history[idx]['created_at']
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get similar queries: {e}")
            return []
    
    def suggest_follow_up_questions(self, query: str, results: List[Dict[str, Any]]) -> List[str]:
        """
        Suggest follow-up questions based on the query and results.
        
        Args:
            query: Original query
            results: Search results
            
        Returns:
            List of suggested follow-up questions
        """
        try:
            suggestions = []
            
            # Extract key terms from query
            query_terms = set(query.lower().split())
            
            # Analyze results for additional topics
            all_text = " ".join([result.get('chunk_text', '') for result in results])
            
            # Simple follow-up suggestions (can be enhanced with more sophisticated methods)
            if 'what' in query.lower():
                suggestions.extend([
                    "Can you provide more details about this topic?",
                    "What are the implications of this information?",
                    "How does this relate to other concepts?"
                ])
            elif 'how' in query.lower():
                suggestions.extend([
                    "What are the steps involved?",
                    "What are the requirements for this process?",
                    "What are the challenges or limitations?"
                ])
            elif 'why' in query.lower():
                suggestions.extend([
                    "What are the underlying causes?",
                    "What are the consequences?",
                    "How does this compare to alternatives?"
                ])
            else:
                suggestions.extend([
                    "Can you explain this in more detail?",
                    "What are the key points to remember?",
                    "How can this be applied in practice?"
                ])
            
            # Add context-specific suggestions based on results
            if results:
                file_types = set()
                for result in results:
                    if result.get('file_name'):
                        file_ext = result['file_name'].split('.')[-1].lower()
                        file_types.add(file_ext)
                
                if 'pdf' in file_types:
                    suggestions.append("Are there any figures or diagrams in the PDFs that might be relevant?")
                
                if len(results) > 3:
                    suggestions.append("Would you like me to focus on a specific aspect of this topic?")
            
            return suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            logger.error(f"Failed to suggest follow-up questions: {e}")
            return []
    
    def get_query_analytics(self) -> Dict[str, Any]:
        """
        Get analytics about queries and usage.
        
        Returns:
            Dictionary containing query analytics
        """
        try:
            # Get query history
            query_history = self.document_store.get_query_history(limit=1000)
            
            if not query_history:
                return {
                    'total_queries': 0,
                    'average_query_length': 0,
                    'most_common_terms': [],
                    'query_trends': []
                }
            
            # Calculate analytics
            total_queries = len(query_history)
            query_lengths = [len(q['query_text']) for q in query_history]
            average_query_length = sum(query_lengths) / len(query_lengths) if query_lengths else 0
            
            # Extract common terms
            all_queries = " ".join([q['query_text'].lower() for q in query_history])
            words = all_queries.split()
            word_counts = {}
            for word in words:
                if len(word) > 3:  # Filter out short words
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            most_common_terms = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'total_queries': total_queries,
                'average_query_length': round(average_query_length, 2),
                'most_common_terms': [{'term': term, 'count': count} for term, count in most_common_terms],
                'query_trends': query_history[-10:]  # Last 10 queries
            }
            
        except Exception as e:
            logger.error(f"Failed to get query analytics: {e}")
            return {}
    
    def refine_query(self, original_query: str, feedback: str) -> str:
        """
        Refine a query based on user feedback.
        
        Args:
            original_query: Original query text
            feedback: User feedback on the results
            
        Returns:
            Refined query text
        """
        try:
            # Simple query refinement (can be enhanced with more sophisticated methods)
            refined_query = original_query
            
            # Add context from feedback
            if 'more specific' in feedback.lower():
                refined_query = f"Provide specific details about {original_query}"
            elif 'broader' in feedback.lower():
                refined_query = f"Give a comprehensive overview of {original_query}"
            elif 'examples' in feedback.lower():
                refined_query = f"Show examples of {original_query}"
            elif 'compare' in feedback.lower():
                refined_query = f"Compare different aspects of {original_query}"
            else:
                # General refinement
                refined_query = f"{original_query} {feedback}"
            
            return refined_query
            
        except Exception as e:
            logger.error(f"Failed to refine query: {e}")
            return original_query
