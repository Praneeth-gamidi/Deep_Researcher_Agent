"""
Advanced usage example for Deep Researcher Agent.
"""

import os
import sys
import json

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deep_researcher import DeepResearcher


def main():
    """Advanced usage example."""
    print("Deep Researcher Agent - Advanced Usage Example")
    print("=" * 60)
    
    # Initialize the researcher with custom settings
    print("Initializing Deep Researcher with custom settings...")
    researcher = DeepResearcher(
        model_name="all-MiniLM-L6-v2",
        data_dir="advanced_example_data",
        chunk_size=800,
        chunk_overlap=150
    )
    
    # Load existing state if available
    if researcher.load_state():
        print("Loaded existing knowledge base state")
    else:
        print("Starting with empty knowledge base")
    
    # Create a comprehensive research document
    print("\nCreating comprehensive research document...")
    
    research_content = """
    Comprehensive Guide to Artificial Intelligence and Machine Learning
    
    Table of Contents:
    1. Introduction to Artificial Intelligence
    2. Machine Learning Fundamentals
    3. Deep Learning and Neural Networks
    4. Natural Language Processing
    5. Computer Vision
    6. Reinforcement Learning
    7. AI Ethics and Responsible AI
    8. Future Directions and Challenges
    
    1. Introduction to Artificial Intelligence
    
    Artificial Intelligence (AI) represents one of the most transformative technologies 
    of the 21st century. It encompasses the development of computer systems capable of 
    performing tasks that traditionally require human intelligence, including learning, 
    reasoning, problem-solving, perception, and language understanding.
    
    The history of AI dates back to the 1950s when Alan Turing proposed the famous 
    Turing Test as a measure of machine intelligence. Since then, AI has evolved through 
    several phases, from early symbolic AI systems to modern deep learning approaches.
    
    Key milestones in AI development include:
    - 1950: Turing Test proposed
    - 1956: Dartmouth Conference and the birth of AI as a field
    - 1960s-1970s: Early expert systems and rule-based AI
    - 1980s: Machine learning approaches gain traction
    - 1990s: Statistical methods and data-driven approaches
    - 2000s: Big data and improved algorithms
    - 2010s: Deep learning revolution
    - 2020s: Large language models and generative AI
    
    2. Machine Learning Fundamentals
    
    Machine Learning (ML) is a subset of AI that focuses on algorithms and statistical 
    models that enable computers to improve their performance on specific tasks through 
    experience, without being explicitly programmed for every scenario.
    
    The machine learning process typically involves:
    
    a) Data Collection and Preprocessing:
    - Gathering relevant datasets
    - Cleaning and validating data
    - Handling missing values and outliers
    - Feature engineering and selection
    
    b) Model Selection and Training:
    - Choosing appropriate algorithms
    - Splitting data into training, validation, and test sets
    - Training models on historical data
    - Tuning hyperparameters for optimal performance
    
    c) Model Evaluation and Validation:
    - Using appropriate metrics (accuracy, precision, recall, F1-score)
    - Cross-validation techniques
    - Avoiding overfitting and underfitting
    - Testing on unseen data
    
    d) Model Deployment and Monitoring:
    - Deploying models to production environments
    - Continuous monitoring of performance
    - Retraining when necessary
    - Handling model drift and concept shift
    
    3. Deep Learning and Neural Networks
    
    Deep Learning represents a paradigm shift in machine learning, utilizing artificial 
    neural networks with multiple layers to automatically learn hierarchical representations 
    of data. This approach has achieved remarkable success across various domains.
    
    Neural Network Architecture:
    
    - Input Layer: Receives raw data (images, text, numerical features)
    - Hidden Layers: Multiple layers of neurons that process information
    - Output Layer: Produces final predictions or classifications
    - Activation Functions: Non-linear functions that introduce complexity
    - Weights and Biases: Parameters learned during training
    
    Types of Neural Networks:
    
    a) Feedforward Neural Networks (FNNs):
    - Simplest type with information flowing in one direction
    - Suitable for tabular data and simple classification tasks
    - Can approximate any continuous function (Universal Approximation Theorem)
    
    b) Convolutional Neural Networks (CNNs):
    - Specialized for image and spatial data processing
    - Use convolutional layers to detect local patterns
    - Include pooling layers for dimensionality reduction
    - Achieved breakthrough results in computer vision tasks
    
    c) Recurrent Neural Networks (RNNs):
    - Designed for sequential data (time series, text, speech)
    - Maintain hidden state to remember previous information
    - Include variants like LSTM and GRU for long-term dependencies
    - Applications in language modeling and time series prediction
    
    d) Transformer Networks:
    - Attention-based architecture that revolutionized NLP
    - Self-attention mechanism allows parallel processing
    - Foundation for modern large language models
    - Achieved state-of-the-art results in many NLP tasks
    
    4. Natural Language Processing
    
    Natural Language Processing (NLP) focuses on enabling computers to understand, 
    interpret, and generate human language in a valuable way. It combines computational 
    linguistics with machine learning and deep learning.
    
    Key NLP Tasks:
    
    a) Text Classification:
    - Sentiment analysis
    - Topic classification
    - Spam detection
    - Language identification
    
    b) Named Entity Recognition (NER):
    - Identifying and classifying entities in text
    - Person names, organizations, locations, dates
    - Medical terms, product names, etc.
    
    c) Machine Translation:
    - Automatic translation between languages
    - Statistical and neural approaches
    - Real-time translation systems
    
    d) Question Answering:
    - Extracting answers from text passages
    - Reading comprehension tasks
    - Conversational AI systems
    
    e) Text Generation:
    - Automatic text creation
    - Creative writing assistance
    - Code generation
    - Chatbot responses
    
    Recent advances in NLP have been driven by:
    - Large language models (GPT, BERT, T5, etc.)
    - Transfer learning and pre-trained models
    - Attention mechanisms and transformer architecture
    - Multilingual and cross-lingual approaches
    
    5. Computer Vision
    
    Computer Vision aims to enable computers to interpret and understand visual 
    information from the world, similar to how humans process visual data.
    
    Core Computer Vision Tasks:
    
    a) Image Classification:
    - Assigning labels to entire images
    - Object recognition and categorization
    - Medical image analysis
    - Quality control in manufacturing
    
    b) Object Detection:
    - Locating and identifying multiple objects in images
    - Bounding box prediction
    - Real-time detection systems
    - Autonomous vehicle perception
    
    c) Image Segmentation:
    - Pixel-level classification
    - Semantic segmentation (grouping similar pixels)
    - Instance segmentation (separating individual objects)
    - Medical image analysis and diagnosis
    
    d) Image Generation:
    - Creating new images from descriptions
    - Style transfer and artistic applications
    - Data augmentation for training
    - Creative and entertainment applications
    
    Key techniques in computer vision include:
    - Convolutional Neural Networks (CNNs)
    - Transfer learning with pre-trained models
    - Data augmentation techniques
    - Object detection frameworks (YOLO, R-CNN, etc.)
    - Generative models (GANs, VAEs, Diffusion models)
    
    6. Reinforcement Learning
    
    Reinforcement Learning (RL) is a type of machine learning where an agent learns 
    to make decisions by interacting with an environment and receiving feedback in 
    the form of rewards or penalties.
    
    Key Components of RL:
    
    a) Agent: The decision-making entity
    b) Environment: The world in which the agent operates
    c) State: Current situation or observation
    d) Action: Decision made by the agent
    e) Reward: Feedback signal from the environment
    f) Policy: Strategy for selecting actions
    
    Types of Reinforcement Learning:
    
    a) Model-Based RL:
    - Agent learns a model of the environment
    - Uses the model to plan actions
    - More sample-efficient but requires accurate models
    
    b) Model-Free RL:
    - Agent learns directly from experience
    - No explicit model of the environment
    - More flexible but may require more data
    
    c) Value-Based Methods:
    - Learn value functions (Q-learning, SARSA)
    - Estimate expected future rewards
    - Popular for discrete action spaces
    
    d) Policy-Based Methods:
    - Learn policies directly
    - Policy gradient methods
    - Good for continuous action spaces
    
    e) Actor-Critic Methods:
    - Combine value and policy approaches
    - Actor learns policy, critic learns value function
    - Often more stable and efficient
    
    Applications of RL include:
    - Game playing (AlphaGo, chess engines)
    - Robotics and autonomous systems
    - Trading and finance
    - Resource allocation
    - Personalized recommendations
    
    7. AI Ethics and Responsible AI
    
    As AI systems become more powerful and widespread, ethical considerations become 
    increasingly important. Responsible AI development requires careful attention to 
    fairness, transparency, accountability, and societal impact.
    
    Key Ethical Principles:
    
    a) Fairness and Non-discrimination:
    - Avoiding biased algorithms and datasets
    - Ensuring equal treatment across different groups
    - Regular auditing for discriminatory patterns
    - Diverse and representative training data
    
    b) Transparency and Explainability:
    - Making AI decisions understandable
    - Providing explanations for automated decisions
    - Open communication about AI capabilities and limitations
    - Clear documentation of AI systems
    
    c) Privacy and Data Protection:
    - Protecting personal and sensitive information
    - Implementing privacy-preserving techniques
    - Complying with data protection regulations
    - Minimizing data collection and retention
    
    d) Safety and Security:
    - Ensuring AI systems are robust and reliable
    - Protecting against adversarial attacks
    - Implementing fail-safe mechanisms
    - Regular security assessments
    
    e) Human Agency and Oversight:
    - Maintaining human control over AI systems
    - Ensuring human decision-making authority
    - Providing human oversight and intervention capabilities
    - Respecting human autonomy and dignity
    
    Challenges in AI Ethics:
    - Algorithmic bias and discrimination
    - Lack of transparency in complex models
    - Privacy concerns with data collection
    - Job displacement and economic impact
    - Misuse of AI for malicious purposes
    - Lack of clear regulatory frameworks
    
    8. Future Directions and Challenges
    
    The field of AI continues to evolve rapidly, with several exciting directions 
    and significant challenges ahead.
    
    Emerging Trends:
    
    a) Large Language Models:
    - Increasingly sophisticated text generation
    - Multimodal capabilities (text, images, audio)
    - Few-shot and zero-shot learning
    - Potential for artificial general intelligence
    
    b) Multimodal AI:
    - Systems that can process multiple types of data
    - Vision-language models
    - Audio-visual understanding
    - Cross-modal learning and generation
    
    c) Edge AI and Federated Learning:
    - Running AI models on edge devices
    - Privacy-preserving distributed learning
    - Reduced latency and bandwidth requirements
    - Improved data privacy and security
    
    d) Quantum Machine Learning:
    - Leveraging quantum computing for ML
    - Quantum algorithms for optimization
    - Potential for exponential speedups
    - New approaches to data processing
    
    Key Challenges:
    
    a) Generalization and Robustness:
    - Creating AI that works across diverse domains
    - Handling distribution shift and novel situations
    - Improving out-of-distribution performance
    - Building more robust and reliable systems
    
    b) Interpretability and Explainability:
    - Understanding how complex models make decisions
    - Building inherently interpretable models
    - Providing meaningful explanations
    - Balancing performance with interpretability
    
    c) Data Requirements and Efficiency:
    - Reducing data requirements for training
    - Improving sample efficiency
    - Handling data scarcity in specialized domains
    - Developing more efficient algorithms
    
    d) Safety and Alignment:
    - Ensuring AI systems remain safe and beneficial
    - Aligning AI goals with human values
    - Preventing misuse and malicious applications
    - Developing robust safety mechanisms
    
    e) Societal Impact and Governance:
    - Managing economic disruption
    - Ensuring equitable access to AI benefits
    - Developing appropriate regulatory frameworks
    - Addressing global AI governance challenges
    
    Conclusion
    
    Artificial Intelligence and Machine Learning represent transformative technologies 
    with the potential to revolutionize virtually every aspect of human life. From 
    healthcare and education to transportation and entertainment, AI is already 
    making significant impacts and will continue to shape our future.
    
    However, realizing the full potential of AI requires careful attention to ethical 
    considerations, responsible development practices, and thoughtful governance. 
    As we continue to advance these technologies, it is crucial to ensure they 
    benefit all of humanity while minimizing potential risks and negative consequences.
    
    The future of AI is bright, but it requires collaboration between researchers, 
    developers, policymakers, and society at large to ensure that AI serves as a 
    force for good in the world.
    """
    
    # Save the research document
    os.makedirs("advanced_example_data/documents", exist_ok=True)
    with open("advanced_example_data/documents/ai_comprehensive_guide.txt", "w", encoding="utf-8") as f:
        f.write(research_content)
    
    # Add the document to knowledge base
    print("Adding comprehensive research document...")
    results = researcher.add_documents(["advanced_example_data/documents/ai_comprehensive_guide.txt"])
    
    for result in results:
        if 'error' not in result:
            print(f"  ✓ {result['file_name']} ({result['chunk_count']} chunks)")
        else:
            print(f"  ✗ Error: {result['error']}")
    
    # Advanced query examples
    print("\n" + "="*60)
    print("ADVANCED QUERY EXAMPLES")
    print("="*60)
    
    advanced_queries = [
        "What are the key milestones in AI development and how have they shaped the field?",
        "Compare and contrast different types of neural networks and their applications",
        "Explain the ethical challenges in AI development and how they can be addressed",
        "What are the emerging trends in AI and what challenges do they present?",
        "How has deep learning revolutionized computer vision and natural language processing?"
    ]
    
    for i, query in enumerate(advanced_queries, 1):
        print(f"\nAdvanced Query {i}: {query}")
        print("-" * 50)
        
        # Use complex query for more sophisticated analysis
        result = researcher.complex_query(query, max_steps=4, explain_reasoning=True)
        
        print(f"Final Answer: {result['final_answer']}")
        print(f"Reasoning Steps: {result['total_steps']}")
        
        # Show reasoning process
        if result.get('reasoning_steps'):
            print("\nReasoning Process:")
            for step in result['reasoning_steps']:
                print(f"  Step {step['step_number']}: {step['sub_query']}")
                if step.get('explanation'):
                    print(f"    Explanation: {step['explanation']}")
    
    # Generate comprehensive research reports
    print("\n" + "="*60)
    print("COMPREHENSIVE RESEARCH REPORTS")
    print("="*60)
    
    report_topics = [
        "AI Ethics and Responsible Development",
        "Future of Machine Learning",
        "Computer Vision Applications",
        "Natural Language Processing Advances"
    ]
    
    for topic in report_topics:
        print(f"\nGenerating report: {topic}")
        print("-" * 40)
        
        report = researcher.generate_report(topic, sources=8, format="markdown")
        
        if 'error' in report:
            print(f"Error: {report['error']}")
        else:
            print(f"Report generated successfully!")
            if 'filename' in report:
                print(f"Saved to: {report['filename']}")
    
    # Advanced analytics
    print("\n" + "="*60)
    print("ADVANCED ANALYTICS")
    print("="*60)
    
    # Get detailed statistics
    stats = researcher.get_knowledge_base_stats()
    print(f"Knowledge Base Statistics:")
    print(f"  Documents: {stats.get('total_documents', 0)}")
    print(f"  Chunks: {stats.get('total_chunks', 0)}")
    print(f"  Queries: {stats.get('total_queries', 0)}")
    
    # Query analytics
    if stats.get('query_analytics'):
        analytics = stats['query_analytics']
        print(f"\nQuery Analytics:")
        print(f"  Average query length: {analytics.get('average_query_length', 0)} characters")
        print(f"  Total queries processed: {analytics.get('total_queries', 0)}")
        
        if analytics.get('most_common_terms'):
            print(f"  Most common terms:")
            for term_data in analytics['most_common_terms'][:10]:
                print(f"    • {term_data['term']} ({term_data['count']} times)")
    
    # Export knowledge base
    print(f"\nExporting knowledge base...")
    export_data = researcher.export_knowledge_base(format="json")
    
    with open("advanced_example_data/knowledge_base_export.json", "w", encoding="utf-8") as f:
        f.write(export_data)
    
    print("Knowledge base exported to: advanced_example_data/knowledge_base_export.json")
    
    # Save state
    researcher.save_state()
    print("Knowledge base state saved")
    
    # Clean up
    researcher.close()
    print("\nAdvanced example completed successfully!")


if __name__ == "__main__":
    main()
