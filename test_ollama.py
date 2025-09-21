#!/usr/bin/env python3
"""
Test script for Ollama integration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deep_researcher import DeepResearcher

def test_ollama_integration():
    """Test the Ollama integration."""
    print("Testing Ollama integration...")
    
    try:
        # Initialize with LLM
        researcher = DeepResearcher(
            use_llm=True,
            llm_model='llama3.2:3b'
        )
        
        # Test query without documents (general knowledge)
        print("\n1. Testing general knowledge query...")
        result = researcher.query("What is machine learning?")
        print(f"Response: {result['response'][:200]}...")
        
        # Test complex query
        print("\n2. Testing complex reasoning...")
        complex_result = researcher.complex_query("Explain the differences between supervised and unsupervised learning")
        print(f"Complex Response: {complex_result['final_answer'][:200]}...")
        
        print("\n✅ Ollama integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("Make sure Ollama is running and the model is available.")

if __name__ == "__main__":
    test_ollama_integration()