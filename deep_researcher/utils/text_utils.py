"""
Text processing and utility functions.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import string

logger = logging.getLogger(__name__)


class TextUtils:
    """
    Utility functions for text processing and analysis.
    """
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', '', text)
        
        return text.strip()
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """
        Extract sentences from text.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        if not text:
            return []
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    @staticmethod
    def extract_paragraphs(text: str) -> List[str]:
        """
        Extract paragraphs from text.
        
        Args:
            text: Input text
            
        Returns:
            List of paragraphs
        """
        if not text:
            return []
        
        # Split by double newlines or single newlines with spacing
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    @staticmethod
    def extract_keywords(text: str, min_length: int = 3, max_keywords: int = 20) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            min_length: Minimum keyword length
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of keywords
        """
        if not text:
            return []
        
        # Convert to lowercase and split into words
        words = text.lower().split()
        
        # Remove punctuation and short words
        words = [word.strip(string.punctuation) for word in words]
        words = [word for word in words if len(word) >= min_length]
        
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'must', 'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        words = [word for word in words if word not in stop_words]
        
        # Count word frequencies
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in keywords[:max_keywords]]
    
    @staticmethod
    def calculate_text_similarity(text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using simple word overlap.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        if not text1 or not text2:
            return 0.0
        
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def summarize_text(text: str, max_sentences: int = 3) -> str:
        """
        Simple text summarization by extracting key sentences.
        
        Args:
            text: Input text
            max_sentences: Maximum number of sentences in summary
            
        Returns:
            Summarized text
        """
        if not text:
            return ""
        
        sentences = TextUtils.extract_sentences(text)
        
        if len(sentences) <= max_sentences:
            return text
        
        # Simple scoring based on word frequency
        words = text.lower().split()
        word_counts = {}
        for word in words:
            if len(word) > 3:  # Filter short words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Score sentences
        sentence_scores = []
        for sentence in sentences:
            score = 0
            sentence_words = sentence.lower().split()
            for word in sentence_words:
                if word in word_counts:
                    score += word_counts[word]
            sentence_scores.append((sentence, score))
        
        # Sort by score and take top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sentence for sentence, score in sentence_scores[:max_sentences]]
        
        return " ".join(top_sentences)
    
    @staticmethod
    def extract_entities(text: str) -> Dict[str, List[str]]:
        """
        Extract basic entities from text (simple implementation).
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            'emails': [],
            'urls': [],
            'numbers': [],
            'dates': [],
            'capitalized_words': []
        }
        
        if not text:
            return entities
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['emails'] = re.findall(email_pattern, text)
        
        # Extract URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        entities['urls'] = re.findall(url_pattern, text)
        
        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        entities['numbers'] = re.findall(number_pattern, text)
        
        # Extract dates (simple pattern)
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        entities['dates'] = re.findall(date_pattern, text)
        
        # Extract capitalized words (potential proper nouns)
        capitalized_pattern = r'\b[A-Z][a-z]+\b'
        entities['capitalized_words'] = re.findall(capitalized_pattern, text)
        
        return entities
    
    @staticmethod
    def format_text_for_display(text: str, max_length: int = 500) -> str:
        """
        Format text for display with length limits.
        
        Args:
            text: Input text
            max_length: Maximum length for display
            
        Returns:
            Formatted text
        """
        if not text:
            return ""
        
        if len(text) <= max_length:
            return text
        
        # Truncate and add ellipsis
        truncated = text[:max_length - 3]
        
        # Try to break at word boundary
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  # If we can break at a reasonable point
            truncated = truncated[:last_space]
        
        return truncated + "..."
    
    @staticmethod
    def highlight_terms(text: str, terms: List[str], highlight_char: str = "*") -> str:
        """
        Highlight specific terms in text.
        
        Args:
            text: Input text
            terms: List of terms to highlight
            highlight_char: Character to use for highlighting
            
        Returns:
            Text with highlighted terms
        """
        if not text or not terms:
            return text
        
        highlighted_text = text
        
        for term in terms:
            if term:
                # Case-insensitive highlighting
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                highlighted_text = pattern.sub(f"{highlight_char}{term}{highlight_char}", highlighted_text)
        
        return highlighted_text
    
    @staticmethod
    def extract_citations(text: str) -> List[str]:
        """
        Extract citations from text (simple implementation).
        
        Args:
            text: Input text
            
        Returns:
            List of citations
        """
        if not text:
            return []
        
        # Simple citation patterns
        citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023) or (Author et al., 2023)
            r'\[[^\]]*\d{4}[^\]]*\]',  # [Author, 2023]
            r'\b\d{4}\b',  # Just years
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        return list(set(citations))  # Remove duplicates
    
    @staticmethod
    def calculate_readability_score(text: str) -> float:
        """
        Calculate a simple readability score.
        
        Args:
            text: Input text
            
        Returns:
            Readability score (higher = more readable)
        """
        if not text:
            return 0.0
        
        sentences = TextUtils.extract_sentences(text)
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        # Simple readability based on average sentence length and word length
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple scoring (can be enhanced with more sophisticated algorithms)
        readability = 100 - (avg_sentence_length * 2) - (avg_word_length * 3)
        
        return max(0, min(100, readability))  # Clamp between 0 and 100
