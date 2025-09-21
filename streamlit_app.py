"""
Streamlit web application for Deep Researcher Agent.
"""

import os
import sys
import logging
import streamlit as st
from pathlib import Path
import tempfile
import json
from datetime import datetime

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deep_researcher import DeepResearcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Deep Researcher Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'researcher' not in st.session_state:
    st.session_state.researcher = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

@st.cache_resource
def initialize_researcher():
    """Initialize the DeepResearcher instance with caching."""
    try:
        researcher = DeepResearcher(
            model_name='all-MiniLM-L6-v2',
            data_dir='data',
            chunk_size=1000,
            chunk_overlap=200,
            use_llm=True,
            llm_model='llama3.2:3b'
        )
        researcher.load_state()
        logger.info("DeepResearcher initialized successfully")
        return researcher
    except Exception as e:
        logger.error(f"Failed to initialize DeepResearcher: {e}")
        st.error(f"Failed to initialize DeepResearcher: {e}")
        return None

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🔍 Deep Researcher Agent")
    st.markdown("Your intelligent research assistant with local embedding capabilities")
    
    # Initialize researcher
    if not st.session_state.initialized:
        with st.spinner("Initializing Deep Researcher Agent..."):
            st.session_state.researcher = initialize_researcher()
            st.session_state.initialized = True
    
    if st.session_state.researcher is None:
        st.error("Failed to initialize the Deep Researcher Agent. Please check the logs.")
        return
    
    researcher = st.session_state.researcher
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['txt', 'md', 'pdf', 'docx', 'html'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, MD, DOCX, or HTML files"
        )
        
        if uploaded_files:
            if st.button("Add Documents", type="primary"):
                for uploaded_file in uploaded_files:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Add to knowledge base
                        results = researcher.add_documents([tmp_path])
                        
                        if results and 'error' not in results[0]:
                            st.success(f"✅ Added: {uploaded_file.name} ({results[0].get('chunk_count', 0)} chunks)")
                        else:
                            error_msg = results[0].get('error', 'Unknown error') if results else 'Unknown error'
                            st.error(f"❌ Failed to add {uploaded_file.name}: {error_msg}")
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    finally:
                        # Clean up temp file
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
        
        # Document list
        st.subheader("📋 Documents in Knowledge Base")
        docs = researcher.list_documents(limit=20)
        
        if docs:
            for doc in docs:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"• {doc['file_name']}")
                    st.caption(f"{doc['chunk_count']} chunks")
                with col2:
                    if st.button("🗑️", key=f"del_{doc['id']}", help="Delete document"):
                        if researcher.delete_document(doc['id']):
                            st.success("Document deleted!")
                            st.rerun()
                        else:
                            st.error("Failed to delete document")
        else:
            st.info("No documents uploaded yet")
        
        # Statistics
        st.subheader("📊 Statistics")
        stats = researcher.get_knowledge_base_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", stats.get('total_documents', 0))
            st.metric("Chunks", stats.get('total_chunks', 0))
        with col2:
            st.metric("Queries", stats.get('total_queries', 0))
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔍 Simple Query", "🧠 Complex Query", "📄 Generate Report"])
    
    with tab1:
        st.header("Simple Query")
        st.markdown("Ask questions about your documents or get general knowledge responses.")
        
        query = st.text_area(
            "Enter your question:",
            placeholder="What is deep learning? How does it work?",
            height=100,
            help="Type your question here. If no documents are uploaded, I'll provide general knowledge responses."
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        with col2:
            if st.button("💡 Example Questions", use_container_width=True):
                st.session_state.example_questions = [
                    "What is machine learning?",
                    "How do neural networks work?",
                    "What are the applications of AI?",
                    "Explain deep learning concepts",
                    "What is natural language processing?"
                ]
        
        # Example questions
        if 'example_questions' in st.session_state:
            st.subheader("💡 Example Questions")
            for i, example in enumerate(st.session_state.example_questions):
                if st.button(f"Q{i+1}: {example}", key=f"example_{i}"):
                    st.session_state.query_text = example
                    st.rerun()
        
        if search_button and query:
            with st.spinner("Thinking..."):
                result = researcher.query(query)
                
                # Display response
                st.subheader("🤖 Response")
                
                if result.get('general_knowledge'):
                    st.info("💡 This is a general knowledge response (no documents in knowledge base)")
                
                st.write(result['response'])
                
                # Show sources if available
                if result.get('results'):
                    st.subheader("📚 Sources")
                    for i, res in enumerate(result['results'][:3]):
                        with st.expander(f"Source {i+1}: {res.get('file_name', 'Unknown')} (Score: {res.get('similarity_score', 0):.3f})"):
                            st.write(res['chunk_text'])
                
                # Show follow-up suggestions
                if result.get('results'):
                    suggestions = researcher.suggest_follow_up_questions(query, result['results'])
                    if suggestions:
                        st.subheader("💭 Suggested Follow-up Questions")
                        for suggestion in suggestions[:3]:
                            if st.button(f"💬 {suggestion}", key=f"suggestion_{hash(suggestion)}"):
                                st.session_state.query_text = suggestion
                                st.rerun()
    
    with tab2:
        st.header("Complex Query with Reasoning")
        st.markdown("Ask complex questions that require multi-step reasoning and analysis.")
        
        complex_query = st.text_area(
            "Enter your complex question:",
            placeholder="Compare and contrast different machine learning algorithms and their applications in real-world scenarios.",
            height=100,
            help="Ask complex questions that require reasoning and analysis."
        )
        
        if st.button("🧠 Analyze", type="primary"):
            if complex_query:
                with st.spinner("Analyzing with reasoning..."):
                    result = researcher.complex_query(complex_query)
                    
                    # Display final answer
                    st.subheader("🎯 Final Answer")
                    st.write(result['final_answer'])
                    
                    # Show reasoning steps
                    if result.get('reasoning_steps'):
                        st.subheader("🔍 Reasoning Process")
                        for step in result['reasoning_steps']:
                            with st.expander(f"Step {step['step_number']}: {step['sub_query']}"):
                                st.write(f"**Explanation:** {step.get('explanation', 'N/A')}")
                                st.write(f"**Response:** {step.get('response', 'N/A')}")
            else:
                st.warning("Please enter a complex question.")
    
    with tab3:
        st.header("Generate Research Report")
        st.markdown("Generate comprehensive research reports on any topic.")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            topic = st.text_input(
                "Enter topic for research report:",
                placeholder="Machine Learning in Healthcare",
                help="Enter the topic you want to research"
            )
        
        with col2:
            format_type = st.selectbox(
                "Report format:",
                ["markdown", "html", "txt", "json"],
                help="Choose the output format"
            )
        
        if st.button("📄 Generate Report", type="primary"):
            if topic:
                with st.spinner("Generating research report..."):
                    result = researcher.generate_report(topic, format=format_type)
                    
                    if 'error' in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success("Report generated successfully!")
                        
                        if 'content' in result:
                            st.subheader("📋 Report Content")
                            st.text_area("Report", result['content'], height=400)
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Report",
                                data=result['content'],
                                file_name=f"report_{topic.replace(' ', '_')}.{format_type}",
                                mime="text/plain"
                            )
                        elif 'filename' in result:
                            st.info(f"Report saved to: {result['filename']}")
            else:
                st.warning("Please enter a topic for the report.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Deep Researcher Agent** - Built with Streamlit | "
        "Powered by sentence-transformers and FAISS"
    )

if __name__ == "__main__":
    main()
