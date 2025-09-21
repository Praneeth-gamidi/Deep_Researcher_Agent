"""
Document processing and text extraction module.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import PyPDF2
import docx
import markdown
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes various document formats and extracts text content.
    """
    
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.html', '.htm'}
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single document and extract text content.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing document metadata and processed content
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Extract text based on file type
            text_content = self._extract_text(file_path)
            
            # Split into chunks
            chunks = self._split_into_chunks(text_content)
            
            # Create document metadata
            document_data = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_type': file_path.suffix.lower(),
                'file_size': file_path.stat().st_size,
                'text_content': text_content,
                'chunks': chunks,
                'chunk_count': len(chunks),
                'total_characters': len(text_content)
            }
            
            logger.info(f"Processed document: {file_path.name} ({len(chunks)} chunks)")
            return document_data
            
        except Exception as e:
            logger.error(f"Failed to process document {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'error': str(e),
                'chunks': [],
                'chunk_count': 0
            }
    
    def process_documents_batch(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of processed document data
        """
        results = []
        for file_path in file_paths:
            result = self.process_document(file_path)
            results.append(result)
        return results
    
    def _extract_text(self, file_path: Path) -> str:
        """
        Extract text content from a document based on its file type.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content
        """
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension == '.txt':
                return self._extract_from_txt(file_path)
            elif file_extension == '.md':
                return self._extract_from_markdown(file_path)
            elif file_extension == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_extension == '.docx':
                return self._extract_from_docx(file_path)
            elif file_extension in ['.html', '.htm']:
                return self._extract_from_html(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
        except Exception as e:
            logger.error(f"Failed to extract text from {file_path}: {e}")
            return ""
    
    def _extract_from_txt(self, file_path: Path) -> str:
        """Extract text from a plain text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _extract_from_markdown(self, file_path: Path) -> str:
        """Extract text from a markdown file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Convert markdown to HTML then extract text
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()
    
    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF file."""
        text = ""
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_from_docx(self, file_path: Path) -> str:
        """Extract text from a DOCX file."""
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def _extract_from_html(self, file_path: Path) -> str:
        """Extract text from an HTML file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        return soup.get_text()
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, save current chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        
        # Simple sentence splitting (can be improved with more sophisticated methods)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _get_overlap_text(self, text: str) -> str:
        """Get the last part of text for overlap."""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Find the last complete sentence within overlap range
        sentences = self._split_into_sentences(text)
        overlap_text = ""
        
        for sentence in reversed(sentences):
            if len(overlap_text + sentence) <= self.chunk_overlap:
                overlap_text = sentence + " " + overlap_text
            else:
                break
        
        return overlap_text.strip()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        import re
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        
        return text.strip()
    
    def get_supported_extensions(self) -> set:
        """Get the set of supported file extensions."""
        return self.SUPPORTED_EXTENSIONS.copy()
    
    def is_supported_file(self, file_path: str) -> bool:
        """Check if a file is supported for processing."""
        return Path(file_path).suffix.lower() in self.SUPPORTED_EXTENSIONS
