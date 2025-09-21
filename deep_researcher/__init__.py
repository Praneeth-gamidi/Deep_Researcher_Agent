"""
Deep Researcher Agent - A comprehensive research agent with local embedding capabilities.
"""

from .deep_researcher import DeepResearcher
from .core.embedding_generator import EmbeddingGenerator
from .core.document_processor import DocumentProcessor
from .core.query_handler import QueryHandler
from .core.reasoning_engine import ReasoningEngine
from .storage.vector_store import VectorStore
from .storage.document_store import DocumentStore
from .utils.text_utils import TextUtils
from .utils.export_utils import ExportUtils

__version__ = "1.0.0"
__author__ = "Deep Researcher Team"

__all__ = [
    "DeepResearcher",
    "EmbeddingGenerator",
    "DocumentProcessor", 
    "QueryHandler",
    "ReasoningEngine",
    "VectorStore",
    "DocumentStore",
    "TextUtils",
    "ExportUtils"
]