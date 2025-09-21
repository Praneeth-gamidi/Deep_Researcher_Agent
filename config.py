"""
Configuration settings for Deep Researcher Agent.
"""

import os
from pathlib import Path

# Base configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
DOCUMENTS_DIR = DATA_DIR / "documents"

# Model configuration
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
ALTERNATIVE_MODELS = [
    "all-MiniLM-L6-v2",  # Fast, good quality
    "all-mpnet-base-v2",  # Higher quality, slower
    "paraphrase-multilingual-MiniLM-L12-v2",  # Multilingual
    "sentence-transformers/all-MiniLM-L6-v2"  # Full path
]

# Document processing configuration
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
MAX_CHUNK_SIZE = 2000
MIN_CHUNK_SIZE = 100

# Vector store configuration
VECTOR_STORE_INDEX_TYPE = "flat"  # Options: "flat", "ivf", "hnsw"
VECTOR_STORE_PERSIST_DIR = EMBEDDINGS_DIR

# Query configuration
DEFAULT_TOP_K = 5
MAX_TOP_K = 50
DEFAULT_MAX_STEPS = 5
MAX_REASONING_STEPS = 10

# Export configuration
SUPPORTED_EXPORT_FORMATS = ["markdown", "pdf", "html", "json", "txt"]
DEFAULT_EXPORT_FORMAT = "markdown"

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "deep_researcher.log"

# Web interface configuration
WEB_HOST = "localhost"
WEB_PORT = 8501
WEB_TITLE = "Deep Researcher Agent"
WEB_ICON = "🔍"

# Performance configuration
BATCH_SIZE = 32
MAX_CONCURRENT_DOCUMENTS = 10
CACHE_SIZE = 1000

# File processing configuration
SUPPORTED_FILE_TYPES = {
    '.txt': 'text/plain',
    '.md': 'text/markdown',
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.html': 'text/html',
    '.htm': 'text/html'
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_DOCUMENTS = 10000

# Security configuration
ALLOWED_ORIGINS = ["*"]
MAX_QUERY_LENGTH = 1000
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 3600  # 1 hour

# Database configuration
DATABASE_PATH = DATA_DIR / "documents.db"
BACKUP_INTERVAL = 24 * 3600  # 24 hours
MAX_BACKUP_FILES = 5

# API configuration (if implemented)
API_VERSION = "1.0"
API_PREFIX = "/api/v1"
API_DOCS_URL = "/docs"
API_REDOC_URL = "/redoc"

# Development configuration
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
VERBOSE = os.getenv("VERBOSE", "False").lower() == "true"
TESTING = os.getenv("TESTING", "False").lower() == "true"

# Environment-specific settings
if TESTING:
    DATA_DIR = BASE_DIR / "test_data"
    LOG_LEVEL = "DEBUG"
    MAX_DOCUMENTS = 100
    BATCH_SIZE = 4

if DEBUG:
    LOG_LEVEL = "DEBUG"
    VERBOSE = True

# Create directories if they don't exist
for directory in [DATA_DIR, EMBEDDINGS_DIR, DOCUMENTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
