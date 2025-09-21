"""
Basic tests for Deep Researcher Agent.
"""

import os
import sys
import tempfile
import shutil
import unittest

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_researcher import DeepResearcher


class TestDeepResearcher(unittest.TestCase):
    """Test cases for DeepResearcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.researcher = DeepResearcher(
            model_name="all-MiniLM-L6-v2",
            data_dir=self.test_dir,
            chunk_size=500,
            chunk_overlap=100
        )
    
    def tearDown(self):
        """Clean up after tests."""
        self.researcher.close()
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test DeepResearcher initialization."""
        self.assertIsNotNone(self.researcher)
        self.assertIsNotNone(self.researcher.embedding_generator)
        self.assertIsNotNone(self.researcher.document_processor)
        self.assertIsNotNone(self.researcher.vector_store)
        self.assertIsNotNone(self.researcher.document_store)
        self.assertIsNotNone(self.researcher.query_handler)
        self.assertIsNotNone(self.researcher.reasoning_engine)
    
    def test_add_documents(self):
        """Test adding documents to knowledge base."""
        # Create a test document
        test_content = """
        This is a test document about artificial intelligence.
        It contains information about machine learning and deep learning.
        The document is used for testing the Deep Researcher Agent.
        """
        
        test_file = os.path.join(self.test_dir, "test_document.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)
        
        # Add document to knowledge base
        results = self.researcher.add_documents([test_file])
        
        # Check results
        self.assertEqual(len(results), 1)
        self.assertNotIn('error', results[0])
        self.assertIn('file_name', results[0])
        self.assertIn('chunk_count', results[0])
        self.assertGreater(results[0]['chunk_count'], 0)
    
    def test_query_processing(self):
        """Test query processing."""
        # Add a test document first
        test_content = """
        Machine learning is a subset of artificial intelligence.
        It involves algorithms that can learn from data.
        Deep learning uses neural networks with multiple layers.
        """
        
        test_file = os.path.join(self.test_dir, "ml_document.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)
        
        self.researcher.add_documents([test_file])
        
        # Test simple query
        query = "What is machine learning?"
        result = self.researcher.query(query)
        
        self.assertIn('query', result)
        self.assertIn('response', result)
        self.assertIn('results', result)
        self.assertEqual(result['query'], query)
        self.assertIsInstance(result['results'], list)
    
    def test_complex_query(self):
        """Test complex query processing."""
        # Add test documents
        test_content1 = """
        Supervised learning uses labeled training data.
        Examples include classification and regression tasks.
        """
        
        test_content2 = """
        Unsupervised learning finds patterns in unlabeled data.
        Examples include clustering and dimensionality reduction.
        """
        
        test_file1 = os.path.join(self.test_dir, "supervised.txt")
        test_file2 = os.path.join(self.test_dir, "unsupervised.txt")
        
        with open(test_file1, "w", encoding="utf-8") as f:
            f.write(test_content1)
        
        with open(test_file2, "w", encoding="utf-8") as f:
            f.write(test_content2)
        
        self.researcher.add_documents([test_file1, test_file2])
        
        # Test complex query
        complex_query = "Compare supervised and unsupervised learning"
        result = self.researcher.complex_query(complex_query, max_steps=3)
        
        self.assertIn('original_query', result)
        self.assertIn('final_answer', result)
        self.assertIn('reasoning_steps', result)
        self.assertEqual(result['original_query'], complex_query)
        self.assertIsInstance(result['reasoning_steps'], list)
        self.assertGreater(result['total_steps'], 0)
    
    def test_knowledge_base_stats(self):
        """Test knowledge base statistics."""
        # Add a test document
        test_content = "This is a test document for statistics."
        test_file = os.path.join(self.test_dir, "stats_test.txt")
        
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)
        
        self.researcher.add_documents([test_file])
        
        # Get statistics
        stats = self.researcher.get_knowledge_base_stats()
        
        self.assertIn('total_documents', stats)
        self.assertIn('total_chunks', stats)
        self.assertIn('total_queries', stats)
        self.assertGreaterEqual(stats['total_documents'], 1)
        self.assertGreaterEqual(stats['total_chunks'], 1)
    
    def test_document_listing(self):
        """Test document listing functionality."""
        # Add test documents
        test_files = []
        for i in range(3):
            content = f"This is test document {i+1}."
            test_file = os.path.join(self.test_dir, f"test_{i+1}.txt")
            
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(content)
            
            test_files.append(test_file)
        
        self.researcher.add_documents(test_files)
        
        # List documents
        docs = self.researcher.list_documents()
        
        self.assertIsInstance(docs, list)
        self.assertEqual(len(docs), 3)
        
        for doc in docs:
            self.assertIn('id', doc)
            self.assertIn('file_name', doc)
            self.assertIn('chunk_count', doc)
    
    def test_document_search(self):
        """Test document search functionality."""
        # Add test documents
        test_content1 = "This document is about machine learning algorithms."
        test_content2 = "This document discusses deep learning neural networks."
        
        test_file1 = os.path.join(self.test_dir, "ml_algorithms.txt")
        test_file2 = os.path.join(self.test_dir, "neural_networks.txt")
        
        with open(test_file1, "w", encoding="utf-8") as f:
            f.write(test_content1)
        
        with open(test_file2, "w", encoding="utf-8") as f:
            f.write(test_content2)
        
        self.researcher.add_documents([test_file1, test_file2])
        
        # Search documents
        search_results = self.researcher.search_documents("machine learning")
        
        self.assertIsInstance(search_results, list)
        self.assertGreater(len(search_results), 0)
        
        # Check that results contain relevant documents
        found_ml_doc = any("ml_algorithms" in doc['file_name'] for doc in search_results)
        self.assertTrue(found_ml_doc)
    
    def test_report_generation(self):
        """Test research report generation."""
        # Add test documents
        test_content = """
        Artificial intelligence is transforming various industries.
        Machine learning enables computers to learn from data.
        Deep learning uses neural networks for complex tasks.
        """
        
        test_file = os.path.join(self.test_dir, "ai_overview.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)
        
        self.researcher.add_documents([test_file])
        
        # Generate report
        report = self.researcher.generate_report("AI Overview", format="markdown")
        
        self.assertIn('topic', report)
        self.assertIn('query', report)
        self.assertEqual(report['topic'], "AI Overview")
        self.assertIn('content', report)
        self.assertIsInstance(report['content'], str)
        self.assertGreater(len(report['content']), 0)
    
    def test_similar_queries(self):
        """Test similar query functionality."""
        # Add a test document
        test_content = "Machine learning is a powerful technology."
        test_file = os.path.join(self.test_dir, "ml_doc.txt")
        
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)
        
        self.researcher.add_documents([test_file])
        
        # Make some queries
        self.researcher.query("What is machine learning?")
        self.researcher.query("How does machine learning work?")
        self.researcher.query("What are the benefits of machine learning?")
        
        # Get similar queries
        similar = self.researcher.get_similar_queries("machine learning benefits")
        
        self.assertIsInstance(similar, list)
        # Should find similar queries from history
    
    def test_follow_up_suggestions(self):
        """Test follow-up question suggestions."""
        # Add a test document
        test_content = "Deep learning uses neural networks with multiple layers."
        test_file = os.path.join(self.test_dir, "deep_learning.txt")
        
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)
        
        self.researcher.add_documents([test_file])
        
        # Make a query
        result = self.researcher.query("What is deep learning?")
        
        # Get follow-up suggestions
        suggestions = self.researcher.suggest_follow_up_questions(
            "What is deep learning?", result['results']
        )
        
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        
        for suggestion in suggestions:
            self.assertIsInstance(suggestion, str)
            self.assertGreater(len(suggestion), 0)


if __name__ == '__main__':
    unittest.main()
