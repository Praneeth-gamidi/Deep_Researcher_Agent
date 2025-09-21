# Deep Researcher Agent - Complete Documentation

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Core Components](#core-components)
5. [API Reference](#api-reference)
6. [Usage Examples](#usage-examples)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [Contributing](#contributing)
10. [License](#license)

## Overview

The Deep Researcher Agent is a comprehensive Python-based research system that can search, analyze, and synthesize information using local embeddings without relying on external web APIs. It's designed for researchers, analysts, and knowledge workers who need to process large amounts of textual information efficiently.

### Key Features

- **Local Embedding Generation**: Uses sentence-transformers for local document embedding
- **Multi-format Document Support**: Processes TXT, MD, PDF, DOCX, and HTML files
- **Advanced Query Processing**: Handles both simple and complex queries with reasoning
- **Vector-based Search**: Fast similarity search using FAISS
- **Multi-step Reasoning**: Breaks down complex queries into manageable steps
- **Report Generation**: Creates research reports in multiple formats
- **Interactive Interfaces**: CLI, web interface, and programmatic API
- **Export Capabilities**: Export results to PDF, Markdown, HTML, JSON, and TXT

## Installation

### Prerequisites

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

### Quick Installation

```bash
# Clone the repository
git clone <repository-url>
cd Deep_Researcher_Agent

# Run the installation script
python install.py
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Create data directories
mkdir -p data/embeddings data/documents

# Test installation
python -c "from deep_researcher import DeepResearcher; print('Installation successful!')"
```

### Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev,web]"

# Run tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=deep_researcher
```

## Quick Start

### Basic Usage

```python
from deep_researcher import DeepResearcher

# Initialize the researcher
researcher = DeepResearcher()

# Add documents
researcher.add_documents(["path/to/document1.pdf", "path/to/document2.txt"])

# Query the knowledge base
result = researcher.query("What are the main findings?")
print(result['response'])

# Complex query with reasoning
complex_result = researcher.complex_query("Compare different methodologies")
print(complex_result['final_answer'])

# Generate a report
report = researcher.generate_report("AI Research", format="pdf")
print(f"Report saved to: {report['filename']}")
```

### Command Line Interface

```bash
# Interactive mode
python main.py

# CLI mode
python main.py --mode cli

# Web interface
python main.py --mode web
```

### Web Interface

```bash
# Start web interface
python main.py --mode web

# Access at http://localhost:8501
```

## Core Components

### 1. EmbeddingGenerator

Generates embeddings for text using local sentence-transformers models.

```python
from deep_researcher import EmbeddingGenerator

generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
embedding = generator.generate_embedding("Sample text")
```

**Key Methods:**
- `generate_embedding(text)`: Generate embedding for single text
- `generate_embeddings_batch(texts)`: Generate embeddings for multiple texts
- `compute_similarity(emb1, emb2)`: Compute cosine similarity
- `find_most_similar(query, candidates, top_k)`: Find similar embeddings

### 2. DocumentProcessor

Processes various document formats and extracts text content.

```python
from deep_researcher import DocumentProcessor

processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
doc_data = processor.process_document("document.pdf")
```

**Supported Formats:**
- Plain text (.txt)
- Markdown (.md)
- PDF (.pdf)
- Word documents (.docx)
- HTML (.html, .htm)

**Key Methods:**
- `process_document(file_path)`: Process single document
- `process_documents_batch(file_paths)`: Process multiple documents
- `extract_text(file_path)`: Extract text from document
- `split_into_chunks(text)`: Split text into overlapping chunks

### 3. VectorStore

Manages vector storage and similarity search using FAISS.

```python
from deep_researcher import VectorStore

store = VectorStore(dimension=384, index_type="flat")
vector_ids = store.add_vectors(embeddings, metadata)
results = store.search(query_embedding, k=5)
```

**Key Methods:**
- `add_vectors(vectors, metadata)`: Add vectors to index
- `search(query_vector, k)`: Search for similar vectors
- `save_index(filename)`: Save index to disk
- `load_index(filename)`: Load index from disk

### 4. DocumentStore

Manages document metadata and relationships using SQLite.

```python
from deep_researcher import DocumentStore

store = DocumentStore(db_path="documents.db")
doc_id = store.add_document(document_data)
document = store.get_document(doc_id)
```

**Key Methods:**
- `add_document(doc_data)`: Add document to store
- `get_document(doc_id)`: Retrieve document by ID
- `list_documents(limit, offset)`: List documents with pagination
- `search_documents(query)`: Search documents by content

### 5. QueryHandler

Processes queries and generates responses.

```python
from deep_researcher import QueryHandler

handler = QueryHandler(embedding_generator, vector_store, document_store)
result = handler.process_query("What is machine learning?")
```

**Key Methods:**
- `process_query(query, top_k)`: Process simple query
- `get_similar_queries(query)`: Find similar queries from history
- `suggest_follow_up_questions(query, results)`: Suggest follow-up questions
- `get_query_analytics()`: Get query statistics

### 6. ReasoningEngine

Handles complex queries with multi-step reasoning.

```python
from deep_researcher import ReasoningEngine

engine = ReasoningEngine(query_handler, embedding_generator)
result = engine.process_complex_query("Compare AI approaches", max_steps=5)
```

**Key Methods:**
- `process_complex_query(query, max_steps)`: Process complex query
- `get_reasoning_explanation(steps)`: Get detailed reasoning explanation

## API Reference

### DeepResearcher Class

Main class that orchestrates all components.

#### Constructor

```python
DeepResearcher(
    model_name="all-MiniLM-L6-v2",
    data_dir="data",
    chunk_size=1000,
    chunk_overlap=200
)
```

**Parameters:**
- `model_name` (str): Sentence transformer model name
- `data_dir` (str): Directory for storing data
- `chunk_size` (int): Size of text chunks
- `chunk_overlap` (int): Overlap between chunks

#### Methods

##### add_documents(file_paths)

Add documents to the knowledge base.

```python
results = researcher.add_documents(["doc1.pdf", "doc2.txt"])
```

**Parameters:**
- `file_paths` (Union[str, List[str]]): File path(s) to add

**Returns:**
- `List[Dict[str, Any]]`: Processing results

##### query(query, top_k=5, include_metadata=True)

Query the knowledge base.

```python
result = researcher.query("What is AI?", top_k=3)
```

**Parameters:**
- `query` (str): Query text
- `top_k` (int): Number of top results
- `include_metadata` (bool): Include metadata in results

**Returns:**
- `Dict[str, Any]`: Query results

##### complex_query(query, max_steps=5, explain_reasoning=True)

Process complex query with reasoning.

```python
result = researcher.complex_query("Compare ML approaches", max_steps=3)
```

**Parameters:**
- `query` (str): Complex query text
- `max_steps` (int): Maximum reasoning steps
- `explain_reasoning` (bool): Include reasoning explanations

**Returns:**
- `Dict[str, Any]`: Complex query results

##### generate_report(topic, sources=5, format="markdown")

Generate research report.

```python
report = researcher.generate_report("AI Ethics", format="pdf")
```

**Parameters:**
- `topic` (str): Report topic
- `sources` (int): Number of sources to use
- `format` (str): Output format (markdown, pdf, html, json, txt)

**Returns:**
- `Dict[str, Any]`: Generated report

##### get_knowledge_base_stats()

Get knowledge base statistics.

```python
stats = researcher.get_knowledge_base_stats()
```

**Returns:**
- `Dict[str, Any]`: Statistics dictionary

##### list_documents(limit=100, offset=0)

List documents in knowledge base.

```python
docs = researcher.list_documents(limit=50)
```

**Returns:**
- `List[Dict[str, Any]]`: List of documents

##### search_documents(query, limit=10)

Search documents by content.

```python
docs = researcher.search_documents("machine learning")
```

**Returns:**
- `List[Dict[str, Any]]`: Matching documents

##### export_knowledge_base(format="json")

Export entire knowledge base.

```python
data = researcher.export_knowledge_base(format="json")
```

**Returns:**
- `str`: Exported data

## Usage Examples

### Example 1: Basic Document Processing

```python
from deep_researcher import DeepResearcher

# Initialize
researcher = DeepResearcher()

# Add documents
documents = [
    "research_paper1.pdf",
    "technical_report.docx", 
    "notes.md"
]
results = researcher.add_documents(documents)

# Process results
for result in results:
    if 'error' not in result:
        print(f"✓ {result['file_name']}: {result['chunk_count']} chunks")
    else:
        print(f"✗ Error: {result['error']}")
```

### Example 2: Advanced Query Processing

```python
# Simple query
simple_result = researcher.query("What are the key findings?")

# Complex query with reasoning
complex_query = """
Compare the different machine learning approaches discussed in the documents
and identify which one would be most suitable for image classification tasks.
Explain your reasoning step by step.
"""

complex_result = researcher.complex_query(complex_query, max_steps=4)

print("Final Answer:")
print(complex_result['final_answer'])

print("\nReasoning Steps:")
for step in complex_result['reasoning_steps']:
    print(f"Step {step['step_number']}: {step['sub_query']}")
    print(f"  Response: {step['response'][:200]}...")
```

### Example 3: Research Report Generation

```python
# Generate different types of reports
topics = [
    "Machine Learning Applications",
    "AI Ethics and Responsible Development", 
    "Future of Artificial Intelligence"
]

for topic in topics:
    print(f"Generating report: {topic}")
    
    # Generate in different formats
    for format_type in ["markdown", "pdf", "html"]:
        report = researcher.generate_report(topic, format=format_type)
        
        if 'filename' in report:
            print(f"  {format_type.upper()}: {report['filename']}")
        elif 'content' in report:
            print(f"  {format_type.upper()}: {len(report['content'])} characters")
```

### Example 4: Interactive Research Session

```python
# Start an interactive research session
def interactive_research():
    researcher = DeepResearcher()
    
    # Load existing knowledge base
    researcher.load_state()
    
    print("Interactive Research Assistant")
    print("Type 'quit' to exit, 'help' for commands")
    
    while True:
        user_input = input("\nYour question: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            break
        
        if user_input.lower() == 'help':
            print("Commands: add <file>, stats, list, search <term>, report <topic>")
            continue
        
        # Process query
        result = researcher.query(user_input)
        print(f"\nAnswer: {result['response']}")
        
        # Show sources
        if result['results']:
            print(f"\nSources ({len(result['results'])}):")
            for i, res in enumerate(result['results'][:3], 1):
                print(f"  {i}. {res.get('file_name', 'Unknown')}")
        
        # Suggest follow-ups
        suggestions = researcher.suggest_follow_up_questions(
            user_input, result['results']
        )
        if suggestions:
            print(f"\nSuggested follow-ups:")
            for suggestion in suggestions[:3]:
                print(f"  • {suggestion}")

# Run interactive session
interactive_research()
```

### Example 5: Batch Processing

```python
import os
from pathlib import Path

def process_document_directory(directory_path):
    """Process all documents in a directory."""
    researcher = DeepResearcher()
    
    # Find all supported documents
    supported_extensions = {'.txt', '.md', '.pdf', '.docx', '.html'}
    document_files = []
    
    for file_path in Path(directory_path).rglob('*'):
        if file_path.suffix.lower() in supported_extensions:
            document_files.append(str(file_path))
    
    print(f"Found {len(document_files)} documents to process")
    
    # Process in batches
    batch_size = 10
    for i in range(0, len(document_files), batch_size):
        batch = document_files[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}...")
        
        results = researcher.add_documents(batch)
        
        # Report results
        successful = sum(1 for r in results if 'error' not in r)
        failed = len(results) - successful
        
        print(f"  ✓ {successful} successful, ✗ {failed} failed")
    
    # Save state
    researcher.save_state()
    print("Processing complete!")

# Process documents
process_document_directory("research_documents/")
```

## Configuration

### Environment Variables

```bash
# Set data directory
export DEEP_RESEARCHER_DATA_DIR="/path/to/data"

# Set model name
export DEEP_RESEARCHER_MODEL="all-MiniLM-L6-v2"

# Set log level
export DEEP_RESEARCHER_LOG_LEVEL="INFO"

# Enable debug mode
export DEEP_RESEARCHER_DEBUG="true"
```

### Configuration File

Create a `config.ini` file:

```ini
[general]
data_dir = data
model_name = all-MiniLM-L6-v2
chunk_size = 1000
chunk_overlap = 200

[logging]
level = INFO
file = deep_researcher.log

[vector_store]
index_type = flat
persist_dir = data/embeddings

[web_interface]
host = localhost
port = 8501
title = Deep Researcher Agent
```

### Custom Configuration

```python
from deep_researcher import DeepResearcher
import config

# Use custom configuration
researcher = DeepResearcher(
    model_name=config.DEFAULT_MODEL_NAME,
    data_dir=config.DATA_DIR,
    chunk_size=config.DEFAULT_CHUNK_SIZE,
    chunk_overlap=config.DEFAULT_CHUNK_OVERLAP
)
```

## Troubleshooting

### Common Issues

#### 1. Memory Issues

**Problem:** Out of memory errors when processing large documents.

**Solutions:**
- Reduce `chunk_size` parameter
- Process documents in smaller batches
- Use a smaller embedding model
- Increase system RAM

```python
# Use smaller chunks
researcher = DeepResearcher(chunk_size=500, chunk_overlap=100)

# Process in batches
for batch in document_batches:
    researcher.add_documents(batch)
```

#### 2. Slow Performance

**Problem:** Slow query processing or document indexing.

**Solutions:**
- Use a faster embedding model
- Reduce the number of results (`top_k`)
- Use GPU acceleration if available
- Optimize vector store settings

```python
# Use faster model
researcher = DeepResearcher(model_name="all-MiniLM-L6-v2")

# Reduce search results
result = researcher.query("query", top_k=3)
```

#### 3. Import Errors

**Problem:** Module import errors.

**Solutions:**
- Ensure all dependencies are installed
- Check Python version compatibility
- Verify package installation

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check installation
python -c "import deep_researcher; print('OK')"
```

#### 4. File Format Issues

**Problem:** Unsupported file formats or corrupted files.

**Solutions:**
- Check file format support
- Verify file integrity
- Convert files to supported formats

```python
# Check supported formats
from deep_researcher import DocumentProcessor
processor = DocumentProcessor()
print(processor.get_supported_extensions())
```

#### 5. Database Issues

**Problem:** SQLite database errors or corruption.

**Solutions:**
- Check database file permissions
- Recreate database if corrupted
- Ensure sufficient disk space

```python
# Recreate database
researcher.clear_knowledge_base()
researcher.add_documents(document_files)
```

### Performance Optimization

#### 1. Model Selection

Choose the right embedding model for your needs:

```python
# Fast, good quality (recommended)
researcher = DeepResearcher(model_name="all-MiniLM-L6-v2")

# Higher quality, slower
researcher = DeepResearcher(model_name="all-mpnet-base-v2")

# Multilingual support
researcher = DeepResearcher(model_name="paraphrase-multilingual-MiniLM-L12-v2")
```

#### 2. Chunk Size Optimization

Optimize chunk size for your documents:

```python
# For short documents
researcher = DeepResearcher(chunk_size=500, chunk_overlap=100)

# For long documents
researcher = DeepResearcher(chunk_size=1500, chunk_overlap=300)

# For technical documents
researcher = DeepResearcher(chunk_size=2000, chunk_overlap=400)
```

#### 3. Vector Store Optimization

Choose the right vector store configuration:

```python
# For small datasets (< 10K documents)
vector_store = VectorStore(dimension=384, index_type="flat")

# For medium datasets (10K-100K documents)
vector_store = VectorStore(dimension=384, index_type="ivf")

# For large datasets (> 100K documents)
vector_store = VectorStore(dimension=384, index_type="hnsw")
```

### Debugging

#### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set in environment
export DEEP_RESEARCHER_LOG_LEVEL="DEBUG"
```

#### Check System Resources

```python
import psutil

# Check memory usage
memory = psutil.virtual_memory()
print(f"Memory usage: {memory.percent}%")

# Check disk space
disk = psutil.disk_usage('/')
print(f"Disk usage: {disk.percent}%")
```

#### Monitor Performance

```python
import time

# Time document processing
start_time = time.time()
results = researcher.add_documents(documents)
processing_time = time.time() - start_time
print(f"Processing time: {processing_time:.2f} seconds")

# Time query processing
start_time = time.time()
result = researcher.query("test query")
query_time = time.time() - start_time
print(f"Query time: {query_time:.2f} seconds")
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd Deep_Researcher_Agent

# Install development dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Run linting
flake8 deep_researcher/
black deep_researcher/

# Run type checking
mypy deep_researcher/
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Write comprehensive docstrings
- Add unit tests for new features
- Update documentation

### Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [Full Documentation](DOCUMENTATION.md)
- **Issues**: [GitHub Issues](https://github.com/deepresearcher/agent/issues)
- **Discussions**: [GitHub Discussions](https://github.com/deepresearcher/agent/discussions)
- **Email**: support@deepresearcher.ai

## Changelog

### Version 1.0.0
- Initial release
- Core functionality implemented
- Multiple document format support
- Vector-based search
- Multi-step reasoning
- Report generation
- Web interface
- CLI interface
- Comprehensive documentation
