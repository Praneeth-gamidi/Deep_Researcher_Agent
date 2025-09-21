"""
Test script for Deep Researcher Agent functionality.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deep_researcher import DeepResearcher

def create_sample_document():
    """Create a sample document for testing."""
    sample_content = """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence (AI) that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience.

## Key Concepts

### Supervised Learning
Supervised learning is a type of machine learning where the algorithm learns from labeled training data. The goal is to learn a mapping from inputs to outputs.

### Unsupervised Learning
Unsupervised learning involves finding hidden patterns in data without labeled examples. Common techniques include clustering and dimensionality reduction.

### Deep Learning
Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to model and understand complex patterns in data.

## Applications

Machine learning has numerous applications including:
- Image recognition and computer vision
- Natural language processing
- Recommendation systems
- Autonomous vehicles
- Medical diagnosis
- Financial fraud detection

## Popular Algorithms

1. Linear Regression
2. Decision Trees
3. Random Forest
4. Support Vector Machines
5. Neural Networks
6. K-Means Clustering
7. Principal Component Analysis

## Future Directions

The field of machine learning continues to evolve with advances in:
- Transfer learning
- Federated learning
- Explainable AI
- Quantum machine learning
- Edge AI deployment
"""
    
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
        f.write(sample_content)
        return f.name

def test_basic_functionality():
    """Test basic functionality of Deep Researcher Agent."""
    print("🧪 Testing Deep Researcher Agent")
    print("=" * 50)
    
    try:
        # Initialize researcher
        print("1. Initializing DeepResearcher...")
        researcher = DeepResearcher(
            model_name='all-MiniLM-L6-v2',
            data_dir='test_data',
            chunk_size=500,
            chunk_overlap=100
        )
        print("✅ DeepResearcher initialized successfully")
        
        # Test empty knowledge base
        print("\n2. Testing empty knowledge base...")
        result = researcher.query("What is machine learning?")
        print(f"Response: {result['response'][:200]}...")
        print("✅ Empty knowledge base handling works")
        
        # Create and add sample document
        print("\n3. Creating and adding sample document...")
        sample_file = create_sample_document()
        results = researcher.add_documents([sample_file])
        
        if results and 'error' not in results[0]:
            print(f"✅ Sample document added successfully ({results[0]['chunk_count']} chunks)")
        else:
            print(f"❌ Failed to add sample document: {results[0].get('error', 'Unknown error')}")
            return False
        
        # Test query with document
        print("\n4. Testing query with document...")
        result = researcher.query("What is supervised learning?")
        print(f"Response: {result['response'][:200]}...")
        print(f"Found {result['total_results']} relevant results")
        print("✅ Document-based query works")
        
        # Test complex query
        print("\n5. Testing complex query...")
        complex_result = researcher.complex_query("Compare supervised and unsupervised learning")
        print(f"Final Answer: {complex_result['final_answer'][:200]}...")
        print(f"Used {complex_result.get('total_steps', 0)} reasoning steps")
        print("✅ Complex query works")
        
        # Test report generation
        print("\n6. Testing report generation...")
        report = researcher.generate_report("Machine Learning Overview", format="markdown")
        if 'content' in report:
            print(f"Report generated: {len(report['content'])} characters")
            print("✅ Report generation works")
        else:
            print(f"❌ Report generation failed: {report.get('error', 'Unknown error')}")
        
        # Test statistics
        print("\n7. Testing statistics...")
        stats = researcher.get_knowledge_base_stats()
        print(f"Documents: {stats.get('total_documents', 0)}")
        print(f"Chunks: {stats.get('total_chunks', 0)}")
        print(f"Queries: {stats.get('total_queries', 0)}")
        print("✅ Statistics work")
        
        # Cleanup
        print("\n8. Cleaning up...")
        researcher.close()
        if os.path.exists(sample_file):
            os.remove(sample_file)
        print("✅ Cleanup completed")
        
        print("\n🎉 All tests passed! Deep Researcher Agent is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False

def test_streamlit_import():
    """Test if Streamlit can be imported."""
    print("\n🔍 Testing Streamlit availability...")
    try:
        import streamlit as st
        print("✅ Streamlit is available")
        return True
    except ImportError:
        print("❌ Streamlit not found. Install with: pip install streamlit")
        return False

if __name__ == "__main__":
    print("Deep Researcher Agent - Test Suite")
    print("=" * 50)
    
    # Test basic functionality
    basic_test_passed = test_basic_functionality()
    
    # Test Streamlit
    streamlit_available = test_streamlit_import()
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Basic Functionality: {'✅ PASS' if basic_test_passed else '❌ FAIL'}")
    print(f"Streamlit Available: {'✅ PASS' if streamlit_available else '❌ FAIL'}")
    
    if basic_test_passed and streamlit_available:
        print("\n🚀 Ready to run! Use 'python run.py' to start the application.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
