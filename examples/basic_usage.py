"""
Basic usage example for Deep Researcher Agent.
"""

import os
import sys

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_researcher import DeepResearcher


def main():
    """Basic usage example."""
    print("Deep Researcher Agent - Basic Usage Example")
    print("=" * 50)
    
    # Initialize the researcher
    print("Initializing Deep Researcher...")
    researcher = DeepResearcher(
        model_name="all-MiniLM-L6-v2",
        data_dir="example_data",
        chunk_size=500,
        chunk_overlap=100
    )
    
    # Add some sample documents
    print("\nAdding sample documents...")
    
    # Create sample text files
    sample_docs = [
        {
            "filename": "ai_research.txt",
            "content": """
            Artificial Intelligence Research Overview
            
            Artificial Intelligence (AI) is a branch of computer science that aims to create 
            intelligent machines that can perform tasks that typically require human intelligence. 
            These tasks include learning, reasoning, problem-solving, perception, and language understanding.
            
            Key areas of AI research include:
            - Machine Learning: Algorithms that improve through experience
            - Natural Language Processing: Understanding and generating human language
            - Computer Vision: Interpreting and understanding visual information
            - Robotics: Creating machines that can interact with the physical world
            - Expert Systems: Computer systems that mimic human expertise
            
            Recent advances in AI have been driven by deep learning, a subset of machine learning 
            that uses neural networks with multiple layers to process data. Deep learning has 
            achieved remarkable success in areas such as image recognition, speech processing, 
            and natural language understanding.
            
            The future of AI research focuses on developing more general artificial intelligence 
            that can perform a wide range of tasks across different domains, rather than being 
            specialized for specific applications.
            """
        },
        {
            "filename": "machine_learning.txt",
            "content": """
            Machine Learning Fundamentals
            
            Machine Learning (ML) is a subset of artificial intelligence that focuses on the 
            development of algorithms and statistical models that enable computer systems to 
            improve their performance on a specific task through experience.
            
            Types of Machine Learning:
            1. Supervised Learning: Learning with labeled training data
               - Classification: Predicting discrete categories
               - Regression: Predicting continuous values
            
            2. Unsupervised Learning: Learning from data without labels
               - Clustering: Grouping similar data points
               - Dimensionality Reduction: Reducing the number of features
            
            3. Reinforcement Learning: Learning through interaction with an environment
               - Agent learns to make decisions by receiving rewards or penalties
            
            Common algorithms include:
            - Linear Regression
            - Decision Trees
            - Random Forest
            - Support Vector Machines
            - Neural Networks
            - K-Means Clustering
            
            The machine learning process typically involves:
            1. Data collection and preprocessing
            2. Feature selection and engineering
            3. Model selection and training
            4. Model evaluation and validation
            5. Model deployment and monitoring
            
            Machine learning has applications in various fields including healthcare, finance, 
            marketing, autonomous vehicles, and recommendation systems.
            """
        },
        {
            "filename": "deep_learning.txt",
            "content": """
            Deep Learning and Neural Networks
            
            Deep Learning is a subset of machine learning that uses artificial neural networks 
            with multiple layers (hence "deep") to model and understand complex patterns in data.
            
            Neural Network Architecture:
            - Input Layer: Receives the input data
            - Hidden Layers: Process the data through weighted connections
            - Output Layer: Produces the final prediction or classification
            
            Types of Neural Networks:
            1. Feedforward Neural Networks: Information flows in one direction
            2. Convolutional Neural Networks (CNNs): Specialized for image processing
            3. Recurrent Neural Networks (RNNs): Designed for sequential data
            4. Long Short-Term Memory (LSTM): Advanced RNN for long sequences
            5. Transformer Networks: Attention-based architecture for NLP
            
            Key Concepts:
            - Backpropagation: Algorithm for training neural networks
            - Gradient Descent: Optimization technique for minimizing loss
            - Activation Functions: Non-linear functions applied to neuron outputs
            - Regularization: Techniques to prevent overfitting
            - Dropout: Randomly setting some neurons to zero during training
            
            Deep learning has revolutionized many fields:
            - Computer Vision: Image classification, object detection, segmentation
            - Natural Language Processing: Language translation, text generation
            - Speech Recognition: Voice assistants, transcription
            - Game Playing: AlphaGo, game AI
            - Autonomous Systems: Self-driving cars, robotics
            
            Challenges in deep learning include the need for large amounts of data, 
            computational resources, and the "black box" nature of deep networks.
            """
        }
    ]
    
    # Create sample files
    for doc in sample_docs:
        file_path = os.path.join("example_data", "documents", doc["filename"])
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc["content"])
    
    # Add documents to knowledge base
    doc_paths = [os.path.join("example_data", "documents", doc["filename"]) for doc in sample_docs]
    results = researcher.add_documents(doc_paths)
    
    print(f"Added {len(results)} documents to knowledge base")
    for result in results:
        if 'error' not in result:
            print(f"  ✓ {result['file_name']} ({result['chunk_count']} chunks)")
        else:
            print(f"  ✗ Error: {result['error']}")
    
    # Example queries
    print("\n" + "="*50)
    print("EXAMPLE QUERIES")
    print("="*50)
    
    queries = [
        "What is artificial intelligence?",
        "What are the different types of machine learning?",
        "How do neural networks work?",
        "What are the applications of deep learning?",
        "Compare supervised and unsupervised learning"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        
        result = researcher.query(query, top_k=3)
        print(f"Response: {result['response']}")
        print(f"Sources: {len(result['results'])} relevant chunks found")
    
    # Complex query example
    print("\n" + "="*50)
    print("COMPLEX QUERY EXAMPLE")
    print("="*50)
    
    complex_query = "Compare the different approaches to machine learning and explain which one would be best for image recognition tasks"
    
    print(f"Complex Query: {complex_query}")
    print("-" * 40)
    
    result = researcher.complex_query(complex_query, max_steps=3)
    print(f"Final Answer: {result['final_answer']}")
    print(f"Reasoning Steps: {result['total_steps']}")
    
    # Generate a research report
    print("\n" + "="*50)
    print("RESEARCH REPORT EXAMPLE")
    print("="*50)
    
    report = researcher.generate_report("Machine Learning Applications", format="markdown")
    print(f"Report generated: {report.get('filename', 'content available')}")
    
    # Show statistics
    print("\n" + "="*50)
    print("KNOWLEDGE BASE STATISTICS")
    print("="*50)
    
    stats = researcher.get_knowledge_base_stats()
    print(f"Total Documents: {stats.get('total_documents', 0)}")
    print(f"Total Chunks: {stats.get('total_chunks', 0)}")
    print(f"Total Queries: {stats.get('total_queries', 0)}")
    
    # Clean up
    researcher.close()
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
