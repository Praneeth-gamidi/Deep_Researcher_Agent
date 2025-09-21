"""
Document storage and metadata management.
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentStore:
    """
    Document storage and metadata management using SQLite.
    """
    
    def __init__(self, db_path: str = "data/documents.db"):
        """
        Initialize the document store.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the SQLite database with required tables."""
        try:
            with sqlite3.connect(self.db_path) as connection:
                cursor = connection.cursor()
                
                # Create documents table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT UNIQUE NOT NULL,
                        file_name TEXT NOT NULL,
                        file_type TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        total_characters INTEGER NOT NULL,
                        chunk_count INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                ''')
                
                # Create chunks table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS chunks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_id INTEGER NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        chunk_text TEXT NOT NULL,
                        chunk_length INTEGER NOT NULL,
                        vector_id INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_id) REFERENCES documents (id)
                    )
                ''')
                
                # Create queries table for query history
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS queries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        query_text TEXT NOT NULL,
                        response_text TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT
                    )
                ''')
                
                connection.commit()
                logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _get_connection(self):
        """Get a new database connection for thread safety."""
        return sqlite3.connect(self.db_path)
    
    def add_document(self, document_data: Dict[str, Any]) -> int:
        """
        Add a document to the store.
        
        Args:
            document_data: Document data dictionary
            
        Returns:
            Document ID
        """
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor()
                
                # Insert document
                cursor.execute('''
                    INSERT OR REPLACE INTO documents 
                    (file_path, file_name, file_type, file_size, total_characters, 
                     chunk_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    document_data['file_path'],
                    document_data['file_name'],
                    document_data['file_type'],
                    document_data['file_size'],
                    document_data['total_characters'],
                    document_data['chunk_count'],
                    json.dumps(document_data.get('metadata', {}))
                ))
                
                document_id = cursor.lastrowid
                
                # Insert chunks
                for i, chunk_text in enumerate(document_data.get('chunks', [])):
                    cursor.execute('''
                        INSERT INTO chunks 
                        (document_id, chunk_index, chunk_text, chunk_length)
                        VALUES (?, ?, ?, ?)
                    ''', (document_id, i, chunk_text, len(chunk_text)))
                
                connection.commit()
                logger.info(f"Document added: {document_data['file_name']} (ID: {document_id})")
                return document_id
            
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            raise
    
    def get_document(self, document_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a document by its ID.
        
        Args:
            document_id: Document ID
            
        Returns:
            Document data dictionary or None
        """
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor()
                
                cursor.execute('''
                    SELECT * FROM documents WHERE id = ?
                ''', (document_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Get column names
                column_names = [description[0] for description in cursor.description]
                
                # Create document dictionary
                document = dict(zip(column_names, row))
                
                # Parse metadata JSON
                if document['metadata']:
                    document['metadata'] = json.loads(document['metadata'])
                else:
                    document['metadata'] = {}
                
                # Get chunks
                cursor.execute('''
                    SELECT chunk_text, chunk_index FROM chunks 
                    WHERE document_id = ? 
                    ORDER BY chunk_index
                ''', (document_id,))
                
                chunks = [row[0] for row in cursor.fetchall()]
                document['chunks'] = chunks
                
                return document
            
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    def get_document_by_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by its file path.
        
        Args:
            file_path: File path
            
        Returns:
            Document data dictionary or None
        """
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor()
                
                cursor.execute('''
                    SELECT id FROM documents WHERE file_path = ?
                ''', (file_path,))
                
                row = cursor.fetchone()
                if row:
                    return self.get_document(row[0])
                return None
            
        except Exception as e:
            logger.error(f"Failed to get document by path {file_path}: {e}")
            return None
    
    def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List documents with pagination.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of document data dictionaries
        """
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor()
                
                cursor.execute('''
                    SELECT id, file_path, file_name, file_type, file_size, 
                           total_characters, chunk_count, created_at, updated_at
                    FROM documents 
                    ORDER BY updated_at DESC 
                    LIMIT ? OFFSET ?
                ''', (limit, offset))
                
                documents = []
                for row in cursor.fetchall():
                    documents.append({
                        'id': row[0],
                        'file_path': row[1],
                        'file_name': row[2],
                        'file_type': row[3],
                        'file_size': row[4],
                        'total_characters': row[5],
                        'chunk_count': row[6],
                        'created_at': row[7],
                        'updated_at': row[8]
                    })
                
                return documents
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search documents by file name or content.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching document data dictionaries
        """
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor()
                
                # Search in file names and chunk content
                cursor.execute('''
                    SELECT DISTINCT d.id, d.file_path, d.file_name, d.file_type, 
                           d.file_size, d.total_characters, d.chunk_count, 
                           d.created_at, d.updated_at
                    FROM documents d
                    LEFT JOIN chunks c ON d.id = c.document_id
                    WHERE d.file_name LIKE ? OR c.chunk_text LIKE ?
                    ORDER BY d.updated_at DESC
                    LIMIT ?
                ''', (f'%{query}%', f'%{query}%', limit))
                
                documents = []
                for row in cursor.fetchall():
                    documents.append({
                        'id': row[0],
                        'file_path': row[1],
                        'file_name': row[2],
                        'file_type': row[3],
                        'file_size': row[4],
                        'total_characters': row[5],
                        'chunk_count': row[6],
                        'created_at': row[7],
                        'updated_at': row[8]
                    })
                
                return documents
            
        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []
    
    def update_document_vector_ids(self, document_id: int, vector_ids: List[int]):
        """
        Update vector IDs for document chunks.
        
        Args:
            document_id: Document ID
            vector_ids: List of vector IDs for each chunk
        """
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor()
                
                for i, vector_id in enumerate(vector_ids):
                    cursor.execute('''
                        UPDATE chunks 
                        SET vector_id = ? 
                        WHERE document_id = ? AND chunk_index = ?
                    ''', (vector_id, document_id, i))
                
                connection.commit()
                logger.info(f"Updated vector IDs for document {document_id}")
            
        except Exception as e:
            logger.error(f"Failed to update vector IDs: {e}")
    
    def get_chunk(self, document_id: int, chunk_index: int) -> Optional[Dict[str, Any]]:
        """
        Get a specific chunk from a document.
        
        Args:
            document_id: Document ID
            chunk_index: Chunk index within the document
            
        Returns:
            Chunk data dictionary or None
        """
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor()
                
                cursor.execute('''
                    SELECT * FROM chunks 
                    WHERE document_id = ? AND chunk_index = ?
                ''', (document_id, chunk_index))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                column_names = [description[0] for description in cursor.description]
                return dict(zip(column_names, row))
            
        except Exception as e:
            logger.error(f"Failed to get chunk: {e}")
            return None
    
    def add_query(self, query_text: str, response_text: str = None, metadata: Dict[str, Any] = None) -> int:
        """
        Add a query to the query history.
        
        Args:
            query_text: Query text
            response_text: Response text
            metadata: Additional metadata
            
        Returns:
            Query ID
        """
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor()
                
                cursor.execute('''
                    INSERT INTO queries (query_text, response_text, metadata)
                    VALUES (?, ?, ?)
                ''', (query_text, response_text, json.dumps(metadata or {})))
                
                query_id = cursor.lastrowid
                connection.commit()
                
                logger.info(f"Query added: {query_id}")
                return query_id
            
        except Exception as e:
            logger.error(f"Failed to add query: {e}")
            raise
    
    def get_query_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get query history.
        
        Args:
            limit: Maximum number of queries to return
            
        Returns:
            List of query data dictionaries
        """
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor()
                
                cursor.execute('''
                    SELECT * FROM queries 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (limit,))
                
                queries = []
                for row in cursor.fetchall():
                    column_names = [description[0] for description in cursor.description]
                    query = dict(zip(column_names, row))
                    
                    # Parse metadata JSON
                    if query['metadata']:
                        query['metadata'] = json.loads(query['metadata'])
                    else:
                        query['metadata'] = {}
                    
                    queries.append(query)
                
                return queries
            
        except Exception as e:
            logger.error(f"Failed to get query history: {e}")
            return []
    
    def delete_document(self, document_id: int) -> bool:
        """
        Delete a document and its chunks.
        
        Args:
            document_id: Document ID
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor()
                
                # Delete chunks first
                cursor.execute('DELETE FROM chunks WHERE document_id = ?', (document_id,))
                
                # Delete document
                cursor.execute('DELETE FROM documents WHERE id = ?', (document_id,))
                
                connection.commit()
                logger.info(f"Document {document_id} deleted")
                return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary containing database statistics
        """
        try:
            with self._get_connection() as connection:
                cursor = connection.cursor()
                
                # Get document count
                cursor.execute('SELECT COUNT(*) FROM documents')
                document_count = cursor.fetchone()[0]
                
                # Get chunk count
                cursor.execute('SELECT COUNT(*) FROM chunks')
                chunk_count = cursor.fetchone()[0]
                
                # Get query count
                cursor.execute('SELECT COUNT(*) FROM queries')
                query_count = cursor.fetchone()[0]
                
                # Get total file size
                cursor.execute('SELECT SUM(file_size) FROM documents')
                total_size = cursor.fetchone()[0] or 0
                
                return {
                    'document_count': document_count,
                    'chunk_count': chunk_count,
                    'query_count': query_count,
                    'total_file_size': total_size,
                    'database_path': self.db_path
                }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
    
    def close(self):
        """Close the database connection."""
        # No longer needed since we use context managers
        logger.info("DocumentStore close called (no persistent connection)")