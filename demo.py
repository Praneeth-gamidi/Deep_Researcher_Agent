"""
Demo script for Deep Researcher Agent.
"""

import os
import sys
import time

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deep_researcher import DeepResearcher


def create_demo_documents():
    """Create demo documents for the demonstration."""
    print("Creating demo documents...")
    
    # Create demo directory
    os.makedirs("demo_data/documents", exist_ok=True)
    
    # Demo document 1: AI Overview
    ai_overview = """
    Artificial Intelligence: A Comprehensive Overview
    
    Artificial Intelligence (AI) represents one of the most transformative technologies 
    of our time. It encompasses the development of computer systems capable of performing 
    tasks that traditionally require human intelligence, including learning, reasoning, 
    problem-solving, perception, and language understanding.
    
    The field of AI has evolved significantly since its inception in the 1950s. Early 
    AI systems were rule-based and limited in scope, but modern AI leverages machine 
    learning algorithms and vast amounts of data to achieve remarkable capabilities.
    
    Key areas of AI include:
    - Machine Learning: Algorithms that improve through experience
    - Natural Language Processing: Understanding and generating human language
    - Computer Vision: Interpreting visual information
    - Robotics: Creating machines that interact with the physical world
    - Expert Systems: Computer systems that mimic human expertise
    
    AI applications are now widespread across industries:
    - Healthcare: Medical diagnosis, drug discovery, personalized treatment
    - Finance: Algorithmic trading, fraud detection, risk assessment
    - Transportation: Autonomous vehicles, traffic optimization
    - Education: Personalized learning, intelligent tutoring systems
    - Entertainment: Recommendation systems, content generation
    
    The future of AI holds great promise, with ongoing research in areas such as 
    artificial general intelligence, quantum machine learning, and brain-computer 
    interfaces. However, it also presents challenges related to ethics, privacy, 
    job displacement, and the need for responsible development.
    """
    
    # Demo document 2: Machine Learning
    machine_learning = """
    Machine Learning: Fundamentals and Applications
    
    Machine Learning (ML) is a subset of artificial intelligence that focuses on the 
    development of algorithms and statistical models that enable computer systems to 
    improve their performance on specific tasks through experience, without being 
    explicitly programmed for every scenario.
    
    Types of Machine Learning:
    
    1. Supervised Learning:
    - Uses labeled training data
    - Examples: Classification, regression
    - Algorithms: Linear regression, decision trees, neural networks
    - Applications: Email spam detection, image recognition, price prediction
    
    2. Unsupervised Learning:
    - Works with unlabeled data
    - Examples: Clustering, dimensionality reduction
    - Algorithms: K-means, hierarchical clustering, PCA
    - Applications: Customer segmentation, anomaly detection, data compression
    
    3. Reinforcement Learning:
    - Learns through interaction with environment
    - Examples: Game playing, robotics, autonomous systems
    - Algorithms: Q-learning, policy gradient methods
    - Applications: Game AI, autonomous vehicles, resource allocation
    
    The machine learning process typically involves:
    1. Data collection and preprocessing
    2. Feature selection and engineering
    3. Model selection and training
    4. Model evaluation and validation
    5. Model deployment and monitoring
    
    Popular machine learning frameworks include:
    - TensorFlow: Google's open-source platform
    - PyTorch: Facebook's dynamic neural network framework
    - Scikit-learn: Python library for traditional ML
    - XGBoost: Gradient boosting framework
    - Keras: High-level neural network API
    
    Machine learning has revolutionized many industries and continues to drive 
    innovation in areas such as healthcare, finance, technology, and scientific research.
    """
    
    # Demo document 3: Deep Learning
    deep_learning = """
    Deep Learning: Neural Networks and Beyond
    
    Deep Learning is a subset of machine learning that uses artificial neural networks 
    with multiple layers (hence "deep") to model and understand complex patterns in data. 
    This approach has achieved remarkable success across various domains and has become 
    the driving force behind many recent AI breakthroughs.
    
    Neural Network Architecture:
    - Input Layer: Receives raw data
    - Hidden Layers: Process information through weighted connections
    - Output Layer: Produces final predictions
    - Activation Functions: Introduce non-linearity
    - Weights and Biases: Learned parameters
    
    Types of Neural Networks:
    
    1. Feedforward Neural Networks (FNNs):
    - Information flows in one direction
    - Suitable for tabular data and simple tasks
    - Can approximate any continuous function
    
    2. Convolutional Neural Networks (CNNs):
    - Specialized for image and spatial data
    - Use convolutional layers for pattern detection
    - Include pooling layers for dimensionality reduction
    - Achieved breakthrough results in computer vision
    
    3. Recurrent Neural Networks (RNNs):
    - Designed for sequential data
    - Maintain hidden state for memory
    - Include LSTM and GRU variants
    - Applications in language modeling and time series
    
    4. Transformer Networks:
    - Attention-based architecture
    - Self-attention mechanism
    - Foundation for large language models
    - Revolutionized natural language processing
    
    Key Concepts in Deep Learning:
    - Backpropagation: Training algorithm
    - Gradient Descent: Optimization technique
    - Activation Functions: ReLU, Sigmoid, Tanh
    - Regularization: Dropout, batch normalization
    - Loss Functions: Cross-entropy, mean squared error
    
    Deep learning has achieved state-of-the-art results in:
    - Computer Vision: Image classification, object detection
    - Natural Language Processing: Translation, text generation
    - Speech Recognition: Voice assistants, transcription
    - Game Playing: AlphaGo, chess engines
    - Scientific Research: Drug discovery, protein folding
    
    Challenges in deep learning include:
    - Need for large amounts of data
    - Computational requirements
    - Interpretability and explainability
    - Overfitting and generalization
    - Ethical considerations and bias
    """
    
    # Save demo documents
    documents = [
        ("ai_overview.txt", ai_overview),
        ("machine_learning.txt", machine_learning),
        ("deep_learning.txt", deep_learning)
    ]
    
    for filename, content in documents:
        filepath = os.path.join("demo_data/documents", filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  ✓ Created {filename}")
    
    return [os.path.join("demo_data/documents", filename) for filename, _ in documents]


def run_demo():
    """Run the complete demonstration."""
    print("Deep Researcher Agent - Interactive Demo")
    print("=" * 50)
    
    # Initialize the researcher
    print("\n1. Initializing Deep Researcher Agent...")
    start_time = time.time()
    
    researcher = DeepResearcher(
        model_name="all-MiniLM-L6-v2",
        data_dir="demo_data",
        chunk_size=800,
        chunk_overlap=150
    )
    
    init_time = time.time() - start_time
    print(f"   ✓ Initialized in {init_time:.2f} seconds")
    
    # Create and add demo documents
    print("\n2. Creating and adding demo documents...")
    doc_start_time = time.time()
    
    document_files = create_demo_documents()
    results = researcher.add_documents(document_files)
    
    doc_time = time.time() - doc_start_time
    print(f"   ✓ Added {len(results)} documents in {doc_time:.2f} seconds")
    
    for result in results:
        if 'error' not in result:
            print(f"     • {result['file_name']}: {result['chunk_count']} chunks")
        else:
            print(f"     ✗ Error: {result['error']}")
    
    # Show knowledge base statistics
    print("\n3. Knowledge Base Statistics:")
    stats = researcher.get_knowledge_base_stats()
    print(f"   • Documents: {stats.get('total_documents', 0)}")
    print(f"   • Chunks: {stats.get('total_chunks', 0)}")
    print(f"   • Vector dimension: {stats.get('vector_store', {}).get('dimension', 'N/A')}")
    
    # Demo queries
    print("\n4. Running Demo Queries:")
    print("-" * 30)
    
    demo_queries = [
        "What is artificial intelligence?",
        "What are the different types of machine learning?",
        "How do neural networks work?",
        "What are the applications of deep learning?",
        "Compare supervised and unsupervised learning"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        
        query_start = time.time()
        result = researcher.query(query, top_k=3)
        query_time = time.time() - query_start
        
        print(f"Response: {result['response'][:200]}{'...' if len(result['response']) > 200 else ''}")
        print(f"Sources: {len(result['results'])} relevant chunks found")
        print(f"Time: {query_time:.2f} seconds")
    
    # Complex query demo
    print("\n5. Complex Query with Reasoning:")
    print("-" * 40)
    
    complex_query = """
    Compare the different approaches to machine learning and deep learning, 
    and explain which approach would be most suitable for a computer vision 
    task like image classification. Provide detailed reasoning for your answer.
    """
    
    print(f"Complex Query: {complex_query}")
    print("\nProcessing with multi-step reasoning...")
    
    complex_start = time.time()
    complex_result = researcher.complex_query(complex_query, max_steps=4, explain_reasoning=True)
    complex_time = time.time() - complex_start
    
    print(f"\nFinal Answer:")
    print(complex_result['final_answer'])
    print(f"\nReasoning Steps: {complex_result['total_steps']}")
    print(f"Processing Time: {complex_time:.2f} seconds")
    
    # Show reasoning process
    if complex_result.get('reasoning_steps'):
        print(f"\nDetailed Reasoning Process:")
        for step in complex_result['reasoning_steps']:
            print(f"\nStep {step['step_number']}: {step['sub_query']}")
            if step.get('explanation'):
                print(f"  Explanation: {step['explanation']}")
            if step.get('response'):
                print(f"  Response: {step['response'][:150]}{'...' if len(step['response']) > 150 else ''}")
    
    # Report generation demo
    print("\n6. Research Report Generation:")
    print("-" * 40)
    
    report_topics = [
        "AI Applications in Healthcare",
        "Future of Machine Learning",
        "Deep Learning Challenges"
    ]
    
    for topic in report_topics:
        print(f"\nGenerating report: '{topic}'")
        
        report_start = time.time()
        report = researcher.generate_report(topic, sources=5, format="markdown")
        report_time = time.time() - report_start
        
        if 'error' in report:
            print(f"  ✗ Error: {report['error']}")
        else:
            print(f"  ✓ Report generated in {report_time:.2f} seconds")
            if 'filename' in report:
                print(f"    Saved to: {report['filename']}")
            elif 'content' in report:
                print(f"    Content length: {len(report['content'])} characters")
    
    # Interactive session
    print("\n7. Interactive Session:")
    print("-" * 40)
    print("Try asking your own questions! (Type 'quit' to exit)")
    
    while True:
        try:
            user_input = input("\nYour question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  help     - Show this help")
                print("  stats    - Show knowledge base statistics")
                print("  list     - List all documents")
                print("  complex  - Ask a complex question")
                print("  quit     - Exit the demo")
                continue
            
            if user_input.lower() == 'stats':
                stats = researcher.get_knowledge_base_stats()
                print(f"\nKnowledge Base Statistics:")
                print(f"  Documents: {stats.get('total_documents', 0)}")
                print(f"  Chunks: {stats.get('total_chunks', 0)}")
                print(f"  Queries: {stats.get('total_queries', 0)}")
                continue
            
            if user_input.lower() == 'list':
                docs = researcher.list_documents()
                print(f"\nDocuments ({len(docs)}):")
                for doc in docs:
                    print(f"  • {doc['file_name']} ({doc['chunk_count']} chunks)")
                continue
            
            if user_input.lower().startswith('complex '):
                complex_question = user_input[8:].strip()
                if complex_question:
                    print(f"\nProcessing complex question: '{complex_question}'")
                    result = researcher.complex_query(complex_question, max_steps=3)
                    print(f"\nAnswer: {result['final_answer']}")
                continue
            
            # Regular query
            print("Thinking...")
            query_start = time.time()
            result = researcher.query(user_input, top_k=3)
            query_time = time.time() - query_start
            
            print(f"\nAnswer: {result['response']}")
            print(f"Sources: {len(result['results'])} relevant chunks")
            print(f"Time: {query_time:.2f} seconds")
            
            # Show sources
            if result['results']:
                print(f"\nTop sources:")
                for i, res in enumerate(result['results'][:3], 1):
                    print(f"  {i}. {res.get('file_name', 'Unknown')}")
            
            # Suggest follow-ups
            suggestions = researcher.suggest_follow_up_questions(user_input, result['results'])
            if suggestions:
                print(f"\nSuggested follow-up questions:")
                for suggestion in suggestions[:3]:
                    print(f"  • {suggestion}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    # Final statistics
    print("\n8. Final Statistics:")
    print("-" * 40)
    
    final_stats = researcher.get_knowledge_base_stats()
    print(f"Total Documents: {final_stats.get('total_documents', 0)}")
    print(f"Total Chunks: {final_stats.get('total_chunks', 0)}")
    print(f"Total Queries: {final_stats.get('total_queries', 0)}")
    
    if final_stats.get('query_analytics'):
        analytics = final_stats['query_analytics']
        print(f"Average Query Length: {analytics.get('average_query_length', 0)} characters")
    
    # Save state
    print("\nSaving knowledge base state...")
    researcher.save_state()
    print("✓ State saved successfully")
    
    # Clean up
    researcher.close()
    print("\nDemo completed successfully!")
    print("\nTo continue using the system:")
    print("1. Run: python main.py")
    print("2. Or run: python examples/basic_usage.py")
    print("3. For web interface: python main.py --mode web")


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nDemo error: {e}")
        print("Please check the error and try again.")
