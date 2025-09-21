"""
Multi-step reasoning engine for complex queries.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import json
from ..core.query_handler import QueryHandler
from ..core.embedding_generator import EmbeddingGenerator
from ..core.llm_generator import LLMGenerator

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """
    Multi-step reasoning engine for breaking down complex queries.
    """
    
    def __init__(self, query_handler: QueryHandler, embedding_generator: EmbeddingGenerator,
                 llm_generator: Optional[LLMGenerator] = None):
        """
        Initialize the reasoning engine.
        
        Args:
            query_handler: Query handler instance
            embedding_generator: Embedding generator instance
            llm_generator: Optional LLM generator for better synthesis
        """
        self.query_handler = query_handler
        self.embedding_generator = embedding_generator
        self.llm_generator = llm_generator
    
    def process_complex_query(self, query: str, max_steps: int = 5, 
                            explain_reasoning: bool = True) -> Dict[str, Any]:
        """
        Process a complex query using multi-step reasoning.
        
        Args:
            query: Complex query to process
            max_steps: Maximum number of reasoning steps
            explain_reasoning: Whether to include reasoning explanations
            
        Returns:
            Dictionary containing reasoning steps and final result
        """
        try:
            # Step 1: Analyze the query and break it down
            analysis = self._analyze_query(query)
            
            # Step 2: Generate sub-queries
            sub_queries = self._generate_sub_queries(query, analysis)
            
            # Step 3: Process each sub-query
            reasoning_steps = []
            all_results = []
            
            for i, sub_query in enumerate(sub_queries[:max_steps]):
                step_result = self._process_reasoning_step(
                    step_number=i + 1,
                    sub_query=sub_query,
                    context=all_results,
                    explain_reasoning=explain_reasoning
                )
                reasoning_steps.append(step_result)
                all_results.extend(step_result.get('results', []))
            
            # Step 4: Synthesize final answer
            final_answer = self._synthesize_answer(query, reasoning_steps, all_results)
            
            return {
                'original_query': query,
                'analysis': analysis,
                'reasoning_steps': reasoning_steps,
                'final_answer': final_answer,
                'total_steps': len(reasoning_steps),
                'total_results': len(all_results)
            }
            
        except Exception as e:
            logger.error(f"Failed to process complex query: {e}")
            return {
                'original_query': query,
                'error': str(e),
                'reasoning_steps': [],
                'final_answer': f"Error processing complex query: {str(e)}"
            }
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query to understand its complexity and requirements.
        
        Args:
            query: Query to analyze
            
        Returns:
            Analysis results
        """
        try:
            analysis = {
                'query_type': self._classify_query_type(query),
                'complexity_score': self._calculate_complexity_score(query),
                'key_concepts': self._extract_key_concepts(query),
                'required_operations': self._identify_required_operations(query),
                'expected_output_type': self._determine_output_type(query)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze query: {e}")
            return {'error': str(e)}
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['compare', 'comparison', 'vs', 'versus', 'difference']):
            return 'comparison'
        elif any(word in query_lower for word in ['analyze', 'analysis', 'examine', 'investigate']):
            return 'analysis'
        elif any(word in query_lower for word in ['explain', 'describe', 'what is', 'define']):
            return 'explanation'
        elif any(word in query_lower for word in ['how to', 'steps', 'process', 'method']):
            return 'how_to'
        elif any(word in query_lower for word in ['why', 'reason', 'cause', 'because']):
            return 'causal'
        elif any(word in query_lower for word in ['summarize', 'summary', 'overview']):
            return 'summary'
        else:
            return 'general'
    
    def _calculate_complexity_score(self, query: str) -> float:
        """Calculate a complexity score for the query."""
        # Simple complexity scoring based on various factors
        score = 0.0
        
        # Length factor
        score += min(len(query.split()) / 20, 1.0) * 0.3
        
        # Question words factor
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        question_count = sum(1 for word in question_words if word in query.lower())
        score += min(question_count / 3, 1.0) * 0.2
        
        # Complex words factor
        complex_words = ['analyze', 'compare', 'evaluate', 'synthesize', 'investigate', 'examine']
        complex_count = sum(1 for word in complex_words if word in query.lower())
        score += min(complex_count / 2, 1.0) * 0.3
        
        # Multiple concepts factor
        concept_indicators = ['and', 'or', 'but', 'however', 'although', 'while']
        concept_count = sum(1 for word in concept_indicators if word in query.lower())
        score += min(concept_count / 3, 1.0) * 0.2
        
        return min(score, 1.0)
    
    def _extract_key_concepts(self, query: str) -> List[str]:
        """Extract key concepts from the query."""
        # Simple concept extraction (can be enhanced with NLP)
        words = query.lower().split()
        
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
        
        concepts = [word for word in words if len(word) > 3 and word not in stop_words]
        
        return concepts[:10]  # Limit to top 10 concepts
    
    def _identify_required_operations(self, query: str) -> List[str]:
        """Identify required operations for the query."""
        operations = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['find', 'search', 'locate']):
            operations.append('search')
        if any(word in query_lower for word in ['compare', 'contrast']):
            operations.append('compare')
        if any(word in query_lower for word in ['analyze', 'examine']):
            operations.append('analyze')
        if any(word in query_lower for word in ['summarize', 'summarise']):
            operations.append('summarize')
        if any(word in query_lower for word in ['explain', 'describe']):
            operations.append('explain')
        if any(word in query_lower for word in ['evaluate', 'assess']):
            operations.append('evaluate')
        
        return operations if operations else ['search']
    
    def _determine_output_type(self, query: str) -> str:
        """Determine the expected output type."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['list', 'enumerate', 'name']):
            return 'list'
        elif any(word in query_lower for word in ['compare', 'comparison']):
            return 'comparison'
        elif any(word in query_lower for word in ['explain', 'describe']):
            return 'explanation'
        elif any(word in query_lower for word in ['summarize', 'summary']):
            return 'summary'
        else:
            return 'general'
    
    def _generate_sub_queries(self, query: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate sub-queries based on the analysis."""
        try:
            sub_queries = []
            query_type = analysis.get('query_type', 'general')
            key_concepts = analysis.get('key_concepts', [])
            operations = analysis.get('required_operations', ['search'])
            
            if query_type == 'comparison':
                # Generate queries for each concept to compare
                for concept in key_concepts[:2]:  # Limit to 2 concepts for comparison
                    sub_queries.append(f"What is {concept}?")
                    sub_queries.append(f"What are the characteristics of {concept}?")
            
            elif query_type == 'analysis':
                # Generate queries for different aspects
                for concept in key_concepts[:3]:
                    sub_queries.append(f"Analyze {concept}")
                    sub_queries.append(f"What are the key aspects of {concept}?")
            
            elif query_type == 'how_to':
                # Generate step-by-step queries
                sub_queries.append(f"What are the prerequisites for {query}?")
                sub_queries.append(f"What are the steps involved in {query}?")
                sub_queries.append(f"What are the best practices for {query}?")
            
            elif query_type == 'causal':
                # Generate cause-effect queries
                sub_queries.append(f"What causes {query.replace('why', '').strip()}?")
                sub_queries.append(f"What are the effects of {query.replace('why', '').strip()}?")
            
            else:
                # General queries
                for concept in key_concepts[:3]:
                    sub_queries.append(f"Explain {concept}")
                    sub_queries.append(f"What is known about {concept}?")
            
            # Add the original query as the final sub-query
            sub_queries.append(query)
            
            return sub_queries[:5]  # Limit to 5 sub-queries
            
        except Exception as e:
            logger.error(f"Failed to generate sub-queries: {e}")
            return [query]
    
    def _process_reasoning_step(self, step_number: int, sub_query: str, 
                              context: List[Dict[str, Any]], 
                              explain_reasoning: bool) -> Dict[str, Any]:
        """Process a single reasoning step."""
        try:
            # Process the sub-query
            result = self.query_handler.process_query(sub_query, top_k=3)
            
            # Create reasoning explanation
            explanation = ""
            if explain_reasoning:
                explanation = self._generate_step_explanation(step_number, sub_query, result, context)
            
            return {
                'step_number': step_number,
                'sub_query': sub_query,
                'results': result.get('results', []),
                'response': result.get('response', ''),
                'explanation': explanation,
                'context_used': len(context)
            }
            
        except Exception as e:
            logger.error(f"Failed to process reasoning step: {e}")
            return {
                'step_number': step_number,
                'sub_query': sub_query,
                'error': str(e),
                'results': [],
                'response': f"Error processing step: {str(e)}"
            }
    
    def _generate_step_explanation(self, step_number: int, sub_query: str, 
                                 result: Dict[str, Any], context: List[Dict[str, Any]]) -> str:
        """Generate explanation for a reasoning step."""
        try:
            explanation_parts = []
            
            explanation_parts.append(f"Step {step_number}: Processing '{sub_query}'")
            
            if result.get('results'):
                explanation_parts.append(f"Found {len(result['results'])} relevant results")
                
                # Mention key sources
                sources = set()
                for res in result['results']:
                    if res.get('file_name'):
                        sources.add(res['file_name'])
                
                if sources:
                    explanation_parts.append(f"Sources: {', '.join(list(sources)[:3])}")
            else:
                explanation_parts.append("No relevant results found")
            
            if context:
                explanation_parts.append(f"Using context from {len(context)} previous steps")
            
            return " | ".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate step explanation: {e}")
            return f"Step {step_number}: Error generating explanation"
    
    def _synthesize_answer(self, original_query: str, reasoning_steps: List[Dict[str, Any]], 
                          all_results: List[Dict[str, Any]]) -> str:
        """Synthesize the final answer from all reasoning steps."""
        try:
            if not reasoning_steps:
                return "Unable to process the query through reasoning steps."
            
            # Use LLM for better synthesis if available
            if self.llm_generator:
                return self.llm_generator.generate_complex_response(original_query, reasoning_steps)
            
            # Fallback to original template-based synthesis
            responses = [step.get('response', '') for step in reasoning_steps if step.get('response')]
            
            if not responses:
                return "No meaningful responses generated from reasoning steps."
            
            synthesis_parts = []
            synthesis_parts.append(f"Based on my analysis of '{original_query}', here's what I found:")
            synthesis_parts.append("")
            
            for i, step in enumerate(reasoning_steps, 1):
                if step.get('response'):
                    synthesis_parts.append(f"Step {i}: {step['response'][:200]}{'...' if len(step['response']) > 200 else ''}")
                    synthesis_parts.append("")
            
            all_sources = set()
            for result in all_results:
                if result.get('file_name'):
                    all_sources.add(result['file_name'])
            
            if all_sources:
                synthesis_parts.append(f"Sources consulted: {', '.join(list(all_sources)[:5])}")
            
            return "\n".join(synthesis_parts)
            
        except Exception as e:
            logger.error(f"Failed to synthesize answer: {e}")
            return f"Error synthesizing final answer: {str(e)}"
    
    def get_reasoning_explanation(self, reasoning_steps: List[Dict[str, Any]]) -> str:
        """Get a detailed explanation of the reasoning process."""
        try:
            if not reasoning_steps:
                return "No reasoning steps to explain."
            
            explanation_parts = []
            explanation_parts.append("Reasoning Process:")
            explanation_parts.append("=" * 50)
            
            for step in reasoning_steps:
                explanation_parts.append(f"\nStep {step['step_number']}: {step['sub_query']}")
                explanation_parts.append("-" * 30)
                
                if step.get('explanation'):
                    explanation_parts.append(f"Explanation: {step['explanation']}")
                
                if step.get('results'):
                    explanation_parts.append(f"Results found: {len(step['results'])}")
                
                if step.get('error'):
                    explanation_parts.append(f"Error: {step['error']}")
            
            return "\n".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"Failed to get reasoning explanation: {e}")
            return f"Error generating reasoning explanation: {str(e)}"
