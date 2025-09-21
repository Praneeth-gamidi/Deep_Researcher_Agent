# Deep Researcher Agent - CodeMate Education Platform

## Quick Start

### Option 1: Web Interface (Recommended)
```bash
python run.py
# Choose option 1 for Streamlit web interface
```

### Option 2: Direct Streamlit
```bash
streamlit run streamlit_app.py
```

### Option 3: Interactive CLI
```bash
python main.py --mode interactive
```

## Features

- 🔍 **Smart Query System**: Ask questions and get intelligent responses
- 📁 **Document Upload**: Support for PDF, TXT, MD, DOCX, HTML files
- 🧠 **Complex Reasoning**: Multi-step analysis and reasoning
- 📄 **Report Generation**: Create research reports in multiple formats
- 💡 **General Knowledge**: Works even without uploaded documents
- 📊 **Real-time Statistics**: Track documents, chunks, and queries

## Usage

1. **Start the application** using one of the methods above
2. **Upload documents** (optional) - PDF, TXT, MD, DOCX, HTML files
3. **Ask questions** - The agent will search through your documents or provide general knowledge
4. **Generate reports** - Create comprehensive research reports on any topic

## Example Questions

- "What is machine learning?"
- "How do neural networks work?"
- "Explain deep learning concepts"
- "What are the applications of AI?"
- "Compare different ML algorithms"

## File Structure

```
Deep_Researcher_Agent/
├── streamlit_app.py          # Main Streamlit web interface
├── main.py                   # CLI and interactive modes
├── run.py                    # Simple startup script
├── deep_researcher/          # Core library
├── requirements.txt          # Dependencies
└── data/                     # Data storage (auto-created)
```

## Requirements

All dependencies are listed in `requirements.txt`. The main ones are:
- streamlit (for web interface)
- sentence-transformers (for embeddings)
- faiss-cpu (for vector search)
- PyPDF2, python-docx (for document processing)

## CodeMate Education Platform

This application is designed to work seamlessly on the CodeMate Education platform:

1. **Web Interface**: Perfect for interactive learning and demonstrations
2. **Local Processing**: All processing happens locally, no external API calls
3. **Educational Focus**: Great for teaching AI, ML, and research concepts
4. **Easy Deployment**: Simple Python application with minimal setup

## Troubleshooting

- If Streamlit doesn't start, try: `pip install streamlit`
- If you get import errors, run: `pip install -r requirements.txt`
- For large files, processing may take a few moments
- The first run will download the embedding model (~90MB)

## Educational Use Cases

- **AI/ML Education**: Demonstrate embedding and vector search concepts
- **Research Skills**: Teach students how to build research assistants
- **Document Analysis**: Show practical applications of NLP
- **Interactive Learning**: Hands-on experience with AI tools
