"""
Main DeepResearcher class that orchestrates all components.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime

from .core.embedding_generator import EmbeddingGenerator
from .core.document_processor import DocumentProcessor
from .core.query_handler import QueryHandler
from .core.reasoning_engine import ReasoningEngine
from .core.llm_generator import LLMGenerator
from .storage.vector_store import VectorStore
from .storage.document_store import DocumentStore
from .utils.text_utils import TextUtils
from .utils.export_utils import ExportUtils

logger = logging.getLogger(__name__)


class DeepResearcher:
    """
    Main DeepResearcher class that orchestrates all components.
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 data_dir: str = "data",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 use_llm: bool = True,
                 llm_model: str = "llama3.2:3b"):
        """
        Initialize the DeepResearcher.
        
        Args:
            model_name: Name of the sentence transformer model
            data_dir: Directory for storing data
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            use_llm: Whether to use LLM for response generation
            llm_model: Ollama model name to use
        """
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Create data directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(data_dir, "embeddings"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "documents"), exist_ok=True)
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator(
            model_name=model_name,
            cache_dir=os.path.join(data_dir, "embeddings")
        )
        
        self.document_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.vector_store = VectorStore(
            dimension=self.embedding_generator.get_embedding_dimension(),
            persist_dir=os.path.join(data_dir, "embeddings")
        )
        
        self.document_store = DocumentStore(
            db_path=os.path.join(data_dir, "documents.db")
        )
        
        # Initialize LLM generator if requested
        self.llm_generator = None
        if use_llm:
            try:
                self.llm_generator = LLMGenerator(model_name=llm_model)
                logger.info(f"LLM generator initialized with model: {llm_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM generator: {e}. Falling back to template responses.")
        
        self.query_handler = QueryHandler(
            embedding_generator=self.embedding_generator,
            vector_store=self.vector_store,
            document_store=self.document_store,
            llm_generator=self.llm_generator
        )
        
        self.reasoning_engine = ReasoningEngine(
            query_handler=self.query_handler,
            embedding_generator=self.embedding_generator,
            llm_generator=self.llm_generator
        )
        
        self.text_utils = TextUtils()
        self.export_utils = ExportUtils()
        
        logger.info("DeepResearcher initialized successfully")
    
    def add_documents(self, file_paths: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Add documents to the knowledge base.
        
        Args:
            file_paths: Single file path or list of file paths
            
        Returns:
            List of processing results
        """
        try:
            if isinstance(file_paths, str):
                file_paths = [file_paths]
            
            results = []
            
            for file_path in file_paths:
                logger.info(f"Processing document: {file_path}")
                
                # Process document
                doc_data = self.document_processor.process_document(file_path)
                
                if 'error' in doc_data:
                    logger.error(f"Failed to process {file_path}: {doc_data['error']}")
                    results.append(doc_data)
                    continue
                
                # Add to document store
                doc_id = self.document_store.add_document(doc_data)
                doc_data['document_id'] = doc_id
                
                # Generate embeddings for chunks
                if doc_data['chunks']:
                    chunk_embeddings = self.embedding_generator.generate_embeddings_batch(doc_data['chunks'])
                    
                    # Prepare metadata for vector store
                    vector_metadata = []
                    for i, chunk in enumerate(doc_data['chunks']):
                        metadata = {
                            'document_id': doc_id,
                            'chunk_index': i,
                            'chunk_text': chunk,
                            'file_name': doc_data['file_name'],
                            'file_path': doc_data['file_path']
                        }
                        vector_metadata.append(metadata)
                    
                    # Add to vector store
                    vector_ids = self.vector_store.add_vectors(
                        np.array(chunk_embeddings),
                        vector_metadata
                    )
                    
                    # Update document store with vector IDs
                    self.document_store.update_document_vector_ids(doc_id, vector_ids)
                
                results.append(doc_data)
                logger.info(f"Successfully processed {file_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return [{'error': str(e)}]
    
    def query(self, query: str, top_k: int = 5, include_metadata: bool = True) -> Dict[str, Any]:
        """
        Query the knowledge base.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            include_metadata: Whether to include metadata
            
        Returns:
            Query results
        """
        try:
            # First try to query the knowledge base
            result = self.query_handler.process_query(query, top_k, include_metadata)
            
            # If no results and knowledge base is empty, provide general response
            if result['total_results'] == 0:
                stats = self.get_knowledge_base_stats()
                if stats.get('total_documents', 0) == 0:
                    result['response'] = self._generate_general_knowledge_response(query)
                    result['general_knowledge'] = True
            
            return result
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            return {
                'query': query,
                'response': f"Error processing query: {str(e)}",
                'results': [],
                'error': str(e)
            }
    
    def _generate_general_knowledge_response(self, query: str) -> str:
        """
        Generate a general knowledge response when no documents are available.
        
        Args:
            query: Query text
            
        Returns:
            General knowledge response
        """
        # This is a simple fallback - in a real implementation, you might use
        # a language model or external API for general knowledge
        response_parts = []
        
        response_parts.append("I don't have any documents in my knowledge base yet, but I can still help!")
        response_parts.append("")
        response_parts.append("Here's what I can tell you about your query:")
        response_parts.append("")
        
        # Simple keyword-based responses for common topics
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['deep learning', 'machine learning', 'ai', 'artificial intelligence']):
            response_parts.append("Deep learning is a subset of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It's particularly effective for tasks like image recognition, natural language processing, and speech recognition.")
            response_parts.append("")
            response_parts.append("Key concepts include:")
            response_parts.append("• Neural networks with multiple hidden layers")
            response_parts.append("• Backpropagation for training")
            response_parts.append("• Convolutional Neural Networks (CNNs) for images")
            response_parts.append("• Recurrent Neural Networks (RNNs) for sequences")
            response_parts.append("• Transformers for natural language processing")
        
        elif any(word in query_lower for word in ['python', 'programming', 'code']):
            response_parts.append("Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in data science, web development, automation, and artificial intelligence.")
            response_parts.append("")
            response_parts.append("Key features include:")
            response_parts.append("• Simple, readable syntax")
            response_parts.append("• Extensive library ecosystem")
            response_parts.append("• Cross-platform compatibility")
            response_parts.append("• Strong community support")
        
        elif any(word in query_lower for word in ['research', 'study', 'analysis']):
            response_parts.append("Research is a systematic investigation to discover new knowledge or validate existing theories. It involves careful planning, data collection, analysis, and interpretation of results.")
            response_parts.append("")
            response_parts.append("Research typically involves:")
            response_parts.append("• Defining research questions")
            response_parts.append("• Literature review")
            response_parts.append("• Methodology design")
            response_parts.append("• Data collection and analysis")
            response_parts.append("• Drawing conclusions and recommendations")
        
        else:
            response_parts.append("That's an interesting question! While I don't have specific documents about this topic in my knowledge base, I'd be happy to help you find information.")
            response_parts.append("")
            response_parts.append("You can:")
            response_parts.append("• Add relevant documents to my knowledge base")
            response_parts.append("• Ask more specific questions")
            response_parts.append("• Use the 'help' command to see what I can do")
        
        response_parts.append("")
        response_parts.append("To get more specific and detailed information, consider adding relevant documents to my knowledge base using the 'add' command!")
        
        return "\n".join(response_parts)
    
    def complex_query(self, query: str, max_steps: int = 5, 
                     explain_reasoning: bool = True) -> Dict[str, Any]:
        """
        Process a complex query using multi-step reasoning.
        
        Args:
            query: Complex query
            max_steps: Maximum reasoning steps
            explain_reasoning: Whether to explain reasoning
            
        Returns:
            Complex query results
        """
        try:
            return self.reasoning_engine.process_complex_query(
                query, max_steps, explain_reasoning
            )
        except Exception as e:
            logger.error(f"Failed to process complex query: {e}")
            return {
                'original_query': query,
                'error': str(e),
                'final_answer': f"Error processing complex query: {str(e)}"
            }
    
    def generate_report(self, topic: str, sources: int = 5, 
                       format: str = "markdown") -> Dict[str, Any]:
        """
        Generate a research report on a topic.
        
        Args:
            topic: Topic to research
            sources: Number of sources to use
            format: Output format (markdown, pdf, html, json, txt)
            
        Returns:
            Generated report
        """
        try:
            # Query for information about the topic
            query_results = self.query(f"Research report on {topic}", top_k=sources)
            
            # Create report structure
            report = {
                'topic': topic,
                'query': f"Research report on {topic}",
                'sources_used': sources,
                'results': query_results.get('results', []),
                'response': query_results.get('response', ''),
                'generated_at': datetime.now().isoformat()
            }
            
            # Export in requested format
            if format.lower() == "markdown":
                report['content'] = self.export_utils.export_to_markdown(report)
            elif format.lower() == "pdf":
                report['filename'] = self.export_utils.export_to_pdf(report)
            elif format.lower() == "html":
                report['content'] = self.export_utils.export_to_html(report)
            elif format.lower() == "json":
                report['content'] = self.export_utils.export_to_json(report)
            elif format.lower() == "txt":
                report['content'] = self.export_utils.export_to_txt(report)
            else:
                report['content'] = self.export_utils.export_to_markdown(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return {
                'topic': topic,
                'error': str(e),
                'content': f"Error generating report: {str(e)}"
            }
    
    def get_similar_queries(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get similar queries from history.
        
        Args:
            query: Query text
            limit: Maximum number of similar queries
            
        Returns:
            List of similar queries
        """
        try:
            return self.query_handler.get_similar_queries(query, limit)
        except Exception as e:
            logger.error(f"Failed to get similar queries: {e}")
            return []
    
    def suggest_follow_up_questions(self, query: str, results: List[Dict[str, Any]]) -> List[str]:
        """
        Suggest follow-up questions.
        
        Args:
            query: Original query
            results: Query results
            
        Returns:
            List of suggested questions
        """
        try:
            return self.query_handler.suggest_follow_up_questions(query, results)
        except Exception as e:
            logger.error(f"Failed to suggest follow-up questions: {e}")
            return []
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        
        Returns:
            Knowledge base statistics
        """
        try:
            vector_stats = self.vector_store.get_index_stats()
            doc_stats = self.document_store.get_stats()
            query_analytics = self.query_handler.get_query_analytics()
            
            return {
                'vector_store': vector_stats,
                'document_store': doc_stats,
                'query_analytics': query_analytics,
                'total_documents': doc_stats.get('document_count', 0),
                'total_chunks': doc_stats.get('chunk_count', 0),
                'total_queries': doc_stats.get('query_count', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get knowledge base stats: {e}")
            return {'error': str(e)}
    
    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List documents in the knowledge base.
        
        Args:
            limit: Maximum number of documents
            offset: Number of documents to skip
            
        Returns:
            List of documents
        """
        try:
            return self.document_store.list_documents(limit, offset)
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents by content.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        try:
            return self.document_store.search_documents(query, limit)
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    def delete_document(self, document_id: int) -> bool:
        """
        Delete a document from the knowledge base.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if successful
        """
        try:
            return self.document_store.delete_document(document_id)
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    def export_knowledge_base(self, format: str = "json") -> str:
        """
        Export the entire knowledge base.
        
        Args:
            format: Export format
            
        Returns:
            Exported data
        """
        try:
            # Get all documents
            documents = self.document_store.list_documents(limit=10000)
            
            # Get all queries
            queries = self.document_store.get_query_history(limit=10000)
            
            # Get statistics
            stats = self.get_knowledge_base_stats()
            
            export_data = {
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'format': format,
                    'version': '1.0'
                },
                'statistics': stats,
                'documents': documents,
                'queries': queries
            }
            
            if format.lower() == "json":
                import json
                return json.dumps(export_data, indent=2, ensure_ascii=False)
            else:
                return str(export_data)
                
        except Exception as e:
            logger.error(f"Failed to export knowledge base: {e}")
            return f"Error exporting knowledge base: {str(e)}"
    
    def save_state(self):
        """Save the current state of the system."""
        try:
            # Save vector store
            self.vector_store.save_index()
            
            # Document store is automatically saved (SQLite)
            logger.info("System state saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def load_state(self) -> bool:
        """Load the saved state of the system."""
        try:
            # Load vector store
            success = self.vector_store.load_index()
            
            if success:
                logger.info("System state loaded successfully")
            else:
                logger.warning("No saved state found or failed to load")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
    
    def clear_knowledge_base(self):
        """Clear the entire knowledge base."""
        try:
            # Clear vector store
            self.vector_store.clear()
            
            # Clear document store
            self.document_store.close()
            os.remove(self.document_store.db_path)
            self.document_store = DocumentStore(db_path=self.document_store.db_path)
            
            logger.info("Knowledge base cleared successfully")
            
        except Exception as e:
            logger.error(f"Failed to clear knowledge base: {e}")
    
    def close(self):
        """Close the system and save state."""
        try:
            self.save_state()
            self.document_store.close()
            logger.info("DeepResearcher closed successfully")
        except Exception as e:
            logger.error(f"Error closing DeepResearcher: {e}")


