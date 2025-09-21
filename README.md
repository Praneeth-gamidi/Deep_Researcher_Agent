# Deep Researcher Agent

A comprehensive research agent with local embedding capabilities, built for educational purposes and CodeMate Education platform deployment.

## ✨ Features

- **🔍 Smart Query System**: Works with or without documents, provides helpful responses
- **📁 Document Upload**: Support for PDF, TXT, MD, DOCX, and HTML files
- **🧠 Complex Reasoning**: Multi-step analysis and reasoning capabilities
- **📄 Report Generation**: Create research reports in multiple formats
- **💡 General Knowledge**: Provides helpful responses even without uploaded documents
- **📊 Real-time Statistics**: Track your knowledge base progress
- **🎨 Beautiful UI**: Clean, modern Streamlit interface
- **🔒 Thread-Safe**: Fixed SQLite thread safety issues for web deployment

## 🚀 Quick Start

### Option 1: Simple Startup (Recommended)
```bash
python start_app.py
```

### Option 2: Direct Streamlit
```bash
streamlit run streamlit_app.py
```

### Option 3: Interactive CLI
```bash
python main.py --mode interactive
```

## 📦 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Deep_Researcher_Agent
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
python start_app.py
```

## 🎯 Usage

### Web Interface (Recommended)

1. **Start the app**: `python start_app.py` and choose option 1
2. **Upload documents**: Use the file upload widget in the sidebar
3. **Ask questions**: Type your questions in the query interface
4. **Generate reports**: Use the report generation tab

### Example Queries

- "What is machine learning?"
- "How do neural networks work?"
- "Compare different AI approaches"
- "Generate a report on deep learning"

### Supported File Formats

- **PDF**: Research papers, documents
- **TXT**: Plain text files
- **MD**: Markdown files
- **DOCX**: Microsoft Word documents
- **HTML**: Web pages

## 🏗️ Architecture

```
Deep_Researcher_Agent/
├── deep_researcher/          # Core library
│   ├── core/                 # Core components
│   │   ├── embedding_generator.py
│   │   ├── document_processor.py
│   │   ├── query_handler.py
│   │   └── reasoning_engine.py
│   ├── storage/              # Storage components
│   │   ├── document_store.py  # Thread-safe SQLite
│   │   └── vector_store.py    # FAISS vector search
│   └── utils/                # Utility functions
├── streamlit_app.py          # Web interface
├── start_app.py              # Simple startup script
├── main.py                   # CLI interface
└── requirements.txt          # Dependencies
```

## 🔧 Recent Fixes

- **✅ Fixed SQLite Thread Safety**: Resolved database connection issues in web interface
- **✅ Improved Query Handling**: Better responses when no documents are uploaded
- **✅ Enhanced Error Handling**: More robust error handling throughout
- **✅ Streamlined UI**: Cleaner, more intuitive user interface
- **✅ Better Performance**: Optimized vector operations and database queries

## 🎓 Educational Use Cases

Perfect for:

- **AI/ML Education**: Demonstrate embedding and vector search concepts
- **Research Skills**: Teach students how to build research assistants
- **Document Analysis**: Show practical applications of NLP
- **Interactive Learning**: Hands-on experience with AI tools
- **CodeMate Platform**: Seamless deployment on educational platforms

## 🌐 CodeMate Education Platform

This application is designed to work seamlessly on the CodeMate Education platform:

1. **Easy Deployment**: Simple Python application with minimal setup
2. **Local Processing**: All processing happens locally, no external API calls
3. **Educational Focus**: Great for teaching AI, ML, and research concepts
4. **Interactive Learning**: Perfect for student engagement and hands-on learning
5. **Thread-Safe**: Fixed database issues for web deployment

## 🛠️ Troubleshooting

### Common Issues

1. **Model Download**: First run downloads the embedding model (~90MB)
2. **Memory Usage**: Large documents may require more RAM
3. **File Formats**: Ensure supported file formats are used
4. **Thread Safety**: Fixed SQLite issues for web interface

### Performance Tips

1. **Chunk Size**: Smaller chunks for better precision, larger for context
2. **Model Choice**: Use smaller models for faster processing
3. **Batch Processing**: Process multiple documents at once

## 📊 Testing

Run the test suite to verify everything works:

```bash
python test_app.py
```

This will test:
- ✅ Basic functionality
- ✅ Document processing
- ✅ Query handling
- ✅ Report generation
- ✅ Streamlit availability

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **sentence-transformers**: For embedding generation
- **FAISS**: For vector similarity search
- **Streamlit**: For the web interface
- **CodeMate Education**: For the platform integration
