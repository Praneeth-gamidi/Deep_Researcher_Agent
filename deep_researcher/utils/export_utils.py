"""
Export utilities for generating reports in various formats.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import markdown
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

logger = logging.getLogger(__name__)


class ExportUtils:
    """
    Utility functions for exporting research results to various formats.
    """
    
    @staticmethod
    def export_to_markdown(results: Dict[str, Any], filename: str = None) -> str:
        """
        Export research results to Markdown format.
        
        Args:
            results: Research results dictionary
            filename: Output filename (optional)
            
        Returns:
            Markdown content as string
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"research_report_{timestamp}.md"
            
            # Create markdown content
            md_content = []
            
            # Title
            md_content.append("# Research Report")
            md_content.append("")
            md_content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            md_content.append("")
            
            # Query
            if 'original_query' in results:
                md_content.append("## Query")
                md_content.append(f"**{results['original_query']}**")
                md_content.append("")
            
            # Analysis
            if 'analysis' in results and results['analysis']:
                md_content.append("## Analysis")
                analysis = results['analysis']
                
                if 'query_type' in analysis:
                    md_content.append(f"**Query Type:** {analysis['query_type']}")
                
                if 'complexity_score' in analysis:
                    md_content.append(f"**Complexity Score:** {analysis['complexity_score']:.2f}")
                
                if 'key_concepts' in analysis:
                    md_content.append(f"**Key Concepts:** {', '.join(analysis['key_concepts'])}")
                
                md_content.append("")
            
            # Reasoning Steps
            if 'reasoning_steps' in results and results['reasoning_steps']:
                md_content.append("## Reasoning Process")
                md_content.append("")
                
                for step in results['reasoning_steps']:
                    md_content.append(f"### Step {step.get('step_number', '?')}")
                    md_content.append(f"**Query:** {step.get('sub_query', 'N/A')}")
                    md_content.append("")
                    
                    if step.get('explanation'):
                        md_content.append(f"**Explanation:** {step['explanation']}")
                        md_content.append("")
                    
                    if step.get('response'):
                        md_content.append("**Response:**")
                        md_content.append(step['response'])
                        md_content.append("")
            
            # Final Answer
            if 'final_answer' in results:
                md_content.append("## Final Answer")
                md_content.append(results['final_answer'])
                md_content.append("")
            
            # Sources
            if 'results' in results and results['results']:
                md_content.append("## Sources")
                md_content.append("")
                
                sources = set()
                for result in results['results']:
                    if result.get('file_name'):
                        sources.add(result['file_name'])
                
                for source in sorted(sources):
                    md_content.append(f"- {source}")
                md_content.append("")
            
            # Statistics
            if 'total_results' in results:
                md_content.append("## Statistics")
                md_content.append(f"- **Total Results:** {results['total_results']}")
                
                if 'total_steps' in results:
                    md_content.append(f"- **Reasoning Steps:** {results['total_steps']}")
                
                md_content.append("")
            
            # Convert to string
            markdown_content = "\n".join(md_content)
            
            # Save to file if filename provided
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                logger.info(f"Markdown report saved to {filename}")
            
            return markdown_content
            
        except Exception as e:
            logger.error(f"Failed to export to markdown: {e}")
            return f"Error generating markdown report: {str(e)}"
    
    @staticmethod
    def export_to_pdf(results: Dict[str, Any], filename: str = None) -> str:
        """
        Export research results to PDF format.
        
        Args:
            results: Research results dictionary
            filename: Output filename (optional)
            
        Returns:
            PDF filename
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"research_report_{timestamp}.pdf"
            
            # Create PDF document
            doc = SimpleDocTemplate(filename, pagesize=A4)
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                spaceBefore=12
            )
            
            # Build content
            content = []
            
            # Title
            content.append(Paragraph("Research Report", title_style))
            content.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            content.append(Spacer(1, 20))
            
            # Query
            if 'original_query' in results:
                content.append(Paragraph("Query", heading_style))
                content.append(Paragraph(results['original_query'], styles['Normal']))
                content.append(Spacer(1, 12))
            
            # Analysis
            if 'analysis' in results and results['analysis']:
                content.append(Paragraph("Analysis", heading_style))
                analysis = results['analysis']
                
                analysis_data = []
                if 'query_type' in analysis:
                    analysis_data.append(['Query Type:', analysis['query_type']])
                if 'complexity_score' in analysis:
                    analysis_data.append(['Complexity Score:', f"{analysis['complexity_score']:.2f}"])
                if 'key_concepts' in analysis:
                    analysis_data.append(['Key Concepts:', ', '.join(analysis['key_concepts'])])
                
                if analysis_data:
                    analysis_table = Table(analysis_data, colWidths=[2*inch, 4*inch])
                    analysis_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 10),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ]))
                    content.append(analysis_table)
                    content.append(Spacer(1, 12))
            
            # Reasoning Steps
            if 'reasoning_steps' in results and results['reasoning_steps']:
                content.append(Paragraph("Reasoning Process", heading_style))
                
                for step in results['reasoning_steps']:
                    step_title = f"Step {step.get('step_number', '?')}"
                    content.append(Paragraph(step_title, styles['Heading3']))
                    
                    if step.get('sub_query'):
                        content.append(Paragraph(f"Query: {step['sub_query']}", styles['Normal']))
                    
                    if step.get('explanation'):
                        content.append(Paragraph(f"Explanation: {step['explanation']}", styles['Normal']))
                    
                    if step.get('response'):
                        content.append(Paragraph("Response:", styles['Heading4']))
                        content.append(Paragraph(step['response'], styles['Normal']))
                    
                    content.append(Spacer(1, 12))
            
            # Final Answer
            if 'final_answer' in results:
                content.append(Paragraph("Final Answer", heading_style))
                content.append(Paragraph(results['final_answer'], styles['Normal']))
                content.append(Spacer(1, 12))
            
            # Sources
            if 'results' in results and results['results']:
                content.append(Paragraph("Sources", heading_style))
                
                sources = set()
                for result in results['results']:
                    if result.get('file_name'):
                        sources.add(result['file_name'])
                
                for source in sorted(sources):
                    content.append(Paragraph(f"• {source}", styles['Normal']))
                
                content.append(Spacer(1, 12))
            
            # Statistics
            if 'total_results' in results:
                content.append(Paragraph("Statistics", heading_style))
                
                stats_data = [['Total Results:', str(results['total_results'])]]
                if 'total_steps' in results:
                    stats_data.append(['Reasoning Steps:', str(results['total_steps'])])
                
                stats_table = Table(stats_data, colWidths=[2*inch, 1*inch])
                stats_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                ]))
                content.append(stats_table)
            
            # Build PDF
            doc.build(content)
            logger.info(f"PDF report saved to {filename}")
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export to PDF: {e}")
            return f"Error generating PDF report: {str(e)}"
    
    @staticmethod
    def export_to_html(results: Dict[str, Any], filename: str = None) -> str:
        """
        Export research results to HTML format.
        
        Args:
            results: Research results dictionary
            filename: Output filename (optional)
            
        Returns:
            HTML content as string
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"research_report_{timestamp}.html"
            
            # Create HTML content
            html_content = []
            
            html_content.append("<!DOCTYPE html>")
            html_content.append("<html lang='en'>")
            html_content.append("<head>")
            html_content.append("    <meta charset='UTF-8'>")
            html_content.append("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
            html_content.append("    <title>Research Report</title>")
            html_content.append("    <style>")
            html_content.append("        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }")
            html_content.append("        h1 { color: #333; border-bottom: 2px solid #333; }")
            html_content.append("        h2 { color: #666; margin-top: 30px; }")
            html_content.append("        h3 { color: #888; }")
            html_content.append("        .meta { color: #666; font-style: italic; }")
            html_content.append("        .step { background-color: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }")
            html_content.append("        .response { background-color: #e8f4f8; padding: 10px; margin: 10px 0; border-left: 4px solid #2196F3; }")
            html_content.append("        .sources { background-color: #f0f8f0; padding: 10px; margin: 10px 0; }")
            html_content.append("    </style>")
            html_content.append("</head>")
            html_content.append("<body>")
            
            # Title
            html_content.append("    <h1>Research Report</h1>")
            html_content.append(f"    <p class='meta'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            
            # Query
            if 'original_query' in results:
                html_content.append("    <h2>Query</h2>")
                html_content.append(f"    <p><strong>{results['original_query']}</strong></p>")
            
            # Analysis
            if 'analysis' in results and results['analysis']:
                html_content.append("    <h2>Analysis</h2>")
                analysis = results['analysis']
                
                if 'query_type' in analysis:
                    html_content.append(f"    <p><strong>Query Type:</strong> {analysis['query_type']}</p>")
                if 'complexity_score' in analysis:
                    html_content.append(f"    <p><strong>Complexity Score:</strong> {analysis['complexity_score']:.2f}</p>")
                if 'key_concepts' in analysis:
                    html_content.append(f"    <p><strong>Key Concepts:</strong> {', '.join(analysis['key_concepts'])}</p>")
            
            # Reasoning Steps
            if 'reasoning_steps' in results and results['reasoning_steps']:
                html_content.append("    <h2>Reasoning Process</h2>")
                
                for step in results['reasoning_steps']:
                    html_content.append(f"    <div class='step'>")
                    html_content.append(f"        <h3>Step {step.get('step_number', '?')}</h3>")
                    
                    if step.get('sub_query'):
                        html_content.append(f"        <p><strong>Query:</strong> {step['sub_query']}</p>")
                    
                    if step.get('explanation'):
                        html_content.append(f"        <p><strong>Explanation:</strong> {step['explanation']}</p>")
                    
                    if step.get('response'):
                        html_content.append(f"        <div class='response'><strong>Response:</strong><br>{step['response']}</div>")
                    
                    html_content.append("    </div>")
            
            # Final Answer
            if 'final_answer' in results:
                html_content.append("    <h2>Final Answer</h2>")
                html_content.append(f"    <div class='response'>{results['final_answer']}</div>")
            
            # Sources
            if 'results' in results and results['results']:
                html_content.append("    <h2>Sources</h2>")
                html_content.append("    <div class='sources'>")
                
                sources = set()
                for result in results['results']:
                    if result.get('file_name'):
                        sources.add(result['file_name'])
                
                for source in sorted(sources):
                    html_content.append(f"        <p>• {source}</p>")
                
                html_content.append("    </div>")
            
            # Statistics
            if 'total_results' in results:
                html_content.append("    <h2>Statistics</h2>")
                html_content.append(f"    <p><strong>Total Results:</strong> {results['total_results']}</p>")
                
                if 'total_steps' in results:
                    html_content.append(f"    <p><strong>Reasoning Steps:</strong> {results['total_steps']}</p>")
            
            html_content.append("</body>")
            html_content.append("</html>")
            
            # Convert to string
            html_string = "\n".join(html_content)
            
            # Save to file if filename provided
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(html_string)
                logger.info(f"HTML report saved to {filename}")
            
            return html_string
            
        except Exception as e:
            logger.error(f"Failed to export to HTML: {e}")
            return f"Error generating HTML report: {str(e)}"
    
    @staticmethod
    def export_to_json(results: Dict[str, Any], filename: str = None) -> str:
        """
        Export research results to JSON format.
        
        Args:
            results: Research results dictionary
            filename: Output filename (optional)
            
        Returns:
            JSON content as string
        """
        try:
            import json
            
            # Add metadata
            export_data = {
                'metadata': {
                    'exported_at': datetime.now().isoformat(),
                    'version': '1.0',
                    'format': 'json'
                },
                'results': results
            }
            
            json_content = json.dumps(export_data, indent=2, ensure_ascii=False)
            
            # Save to file if filename provided
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(json_content)
                logger.info(f"JSON report saved to {filename}")
            
            return json_content
            
        except Exception as e:
            logger.error(f"Failed to export to JSON: {e}")
            return f"Error generating JSON report: {str(e)}"
    
    @staticmethod
    def export_to_txt(results: Dict[str, Any], filename: str = None) -> str:
        """
        Export research results to plain text format.
        
        Args:
            results: Research results dictionary
            filename: Output filename (optional)
            
        Returns:
            Text content as string
        """
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"research_report_{timestamp}.txt"
            
            # Create text content
            text_content = []
            
            text_content.append("RESEARCH REPORT")
            text_content.append("=" * 50)
            text_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            text_content.append("")
            
            # Query
            if 'original_query' in results:
                text_content.append("QUERY:")
                text_content.append("-" * 20)
                text_content.append(results['original_query'])
                text_content.append("")
            
            # Analysis
            if 'analysis' in results and results['analysis']:
                text_content.append("ANALYSIS:")
                text_content.append("-" * 20)
                analysis = results['analysis']
                
                if 'query_type' in analysis:
                    text_content.append(f"Query Type: {analysis['query_type']}")
                if 'complexity_score' in analysis:
                    text_content.append(f"Complexity Score: {analysis['complexity_score']:.2f}")
                if 'key_concepts' in analysis:
                    text_content.append(f"Key Concepts: {', '.join(analysis['key_concepts'])}")
                
                text_content.append("")
            
            # Reasoning Steps
            if 'reasoning_steps' in results and results['reasoning_steps']:
                text_content.append("REASONING PROCESS:")
                text_content.append("-" * 20)
                
                for step in results['reasoning_steps']:
                    text_content.append(f"Step {step.get('step_number', '?')}:")
                    
                    if step.get('sub_query'):
                        text_content.append(f"  Query: {step['sub_query']}")
                    
                    if step.get('explanation'):
                        text_content.append(f"  Explanation: {step['explanation']}")
                    
                    if step.get('response'):
                        text_content.append(f"  Response: {step['response']}")
                    
                    text_content.append("")
            
            # Final Answer
            if 'final_answer' in results:
                text_content.append("FINAL ANSWER:")
                text_content.append("-" * 20)
                text_content.append(results['final_answer'])
                text_content.append("")
            
            # Sources
            if 'results' in results and results['results']:
                text_content.append("SOURCES:")
                text_content.append("-" * 20)
                
                sources = set()
                for result in results['results']:
                    if result.get('file_name'):
                        sources.add(result['file_name'])
                
                for source in sorted(sources):
                    text_content.append(f"• {source}")
                
                text_content.append("")
            
            # Statistics
            if 'total_results' in results:
                text_content.append("STATISTICS:")
                text_content.append("-" * 20)
                text_content.append(f"Total Results: {results['total_results']}")
                
                if 'total_steps' in results:
                    text_content.append(f"Reasoning Steps: {results['total_steps']}")
            
            # Convert to string
            text_string = "\n".join(text_content)
            
            # Save to file if filename provided
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text_string)
                logger.info(f"Text report saved to {filename}")
            
            return text_string
            
        except Exception as e:
            logger.error(f"Failed to export to text: {e}")
            return f"Error generating text report: {str(e)}"
