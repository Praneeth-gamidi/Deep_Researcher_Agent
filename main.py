"""
Main application entry point for Deep Researcher Agent.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deep_researcher import DeepResearcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deep_researcher.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main application function."""
    parser = argparse.ArgumentParser(description='Deep Researcher Agent')
    parser.add_argument('--mode', choices=['cli', 'web', 'interactive'], 
                       default='interactive', help='Application mode')
    parser.add_argument('--data-dir', default='data', 
                       help='Data directory path')
    parser.add_argument('--model', default='all-MiniLM-L6-v2',
                       help='Embedding model name')
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Text chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=200,
                       help='Text chunk overlap')
    
    args = parser.parse_args()
    
    try:
        # Initialize DeepResearcher
        logger.info("Initializing Deep Researcher Agent...")
        researcher = DeepResearcher(
            model_name=args.model,
            data_dir=args.data_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        # Load existing state
        researcher.load_state()
        
        if args.mode == 'cli':
            run_cli_mode(researcher)
        elif args.mode == 'web':
            run_web_mode(researcher)
        elif args.mode == 'interactive':
            run_interactive_mode(researcher)
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        if 'researcher' in locals():
            researcher.close()


def run_cli_mode(researcher):
    """Run in CLI mode."""
    print("Deep Researcher Agent - CLI Mode")
    print("=" * 40)
    print("Commands:")
    print("  add <file_path>     - Add document(s)")
    print("  query <text>        - Query knowledge base")
    print("  complex <text>      - Complex query with reasoning")
    print("  report <topic>      - Generate research report")
    print("  stats               - Show statistics")
    print("  list                - List documents")
    print("  search <text>       - Search documents")
    print("  help                - Show this help")
    print("  quit                - Exit application")
    print()
    
    while True:
        try:
            command = input("> ").strip()
            
            if not command:
                continue
            
            parts = command.split(' ', 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if cmd == 'quit' or cmd == 'exit':
                break
            elif cmd == 'help':
                print("Commands: add, query, complex, report, stats, list, search, help, quit")
            elif cmd == 'add':
                if not arg:
                    print("Usage: add <file_path>")
                    continue
                
                file_paths = [f.strip() for f in arg.split(',')]
                results = researcher.add_documents(file_paths)
                
                for result in results:
                    if 'error' in result:
                        print(f"Error: {result['error']}")
                    else:
                        print(f"Added: {result['file_name']} ({result['chunk_count']} chunks)")
            
            elif cmd == 'query':
                if not arg:
                    print("Usage: query <text>")
                    continue
                
                result = researcher.query(arg)
                print(f"\nResponse: {result['response']}")
                print(f"Found {result['total_results']} results")
            
            elif cmd == 'complex':
                if not arg:
                    print("Usage: complex <text>")
                    continue
                
                result = researcher.complex_query(arg)
                print(f"\nFinal Answer: {result['final_answer']}")
                print(f"Used {result['total_steps']} reasoning steps")
            
            elif cmd == 'report':
                if not arg:
                    print("Usage: report <topic>")
                    continue
                
                result = researcher.generate_report(arg)
                if 'error' in result:
                    print(f"Error: {result['error']}")
                else:
                    print(f"Report generated: {result.get('filename', 'content available')}")
            
            elif cmd == 'stats':
                stats = researcher.get_knowledge_base_stats()
                print(f"\nKnowledge Base Statistics:")
                print(f"Documents: {stats.get('total_documents', 0)}")
                print(f"Chunks: {stats.get('total_chunks', 0)}")
                print(f"Queries: {stats.get('total_queries', 0)}")
            
            elif cmd == 'list':
                docs = researcher.list_documents()
                print(f"\nDocuments ({len(docs)}):")
                for doc in docs:
                    print(f"  {doc['id']}: {doc['file_name']} ({doc['chunk_count']} chunks)")
            
            elif cmd == 'search':
                if not arg:
                    print("Usage: search <text>")
                    continue
                
                docs = researcher.search_documents(arg)
                print(f"\nFound {len(docs)} documents:")
                for doc in docs:
                    print(f"  {doc['id']}: {doc['file_name']}")
            
            else:
                print("Unknown command. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


def run_web_mode(researcher):
    """Run in web mode using Streamlit."""
    try:
        import subprocess
        import sys
        
        print("Starting Streamlit web interface...")
        print("The web interface will open in your browser at http://localhost:8501")
        print("Press Ctrl+C to stop the server")
        
        # Run streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    
    except ImportError:
        print("Streamlit not installed. Please install it with: pip install streamlit")
        print("Then run: streamlit run streamlit_app.py")
    except KeyboardInterrupt:
        print("\nStreamlit server stopped.")
    except Exception as e:
        print(f"Error running Streamlit: {e}")
        print("You can manually run: streamlit run streamlit_app.py")


def run_interactive_mode(researcher):
    """Run in interactive mode."""
    print("Deep Researcher Agent - Interactive Mode")
    print("=" * 50)
    print("Welcome! This is an interactive research assistant.")
    print("You can ask questions, add documents, and generate reports.")
    print("Type 'help' for commands or 'quit' to exit.")
    print()
    
    # Show initial statistics
    stats = researcher.get_knowledge_base_stats()
    print(f"Knowledge Base: {stats.get('total_documents', 0)} documents, "
          f"{stats.get('total_chunks', 0)} chunks")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  help                    - Show this help")
                print("  add <file_path>         - Add document(s)")
                print("  stats                   - Show statistics")
                print("  list                    - List documents")
                print("  search <text>           - Search documents")
                print("  report <topic>          - Generate research report")
                print("  quit/exit/bye           - Exit")
                print("\nOr just ask a question directly!")
                continue
            
            # Check for commands
            if user_input.startswith('add '):
                file_path = user_input[4:].strip()
                if file_path:
                    results = researcher.add_documents([file_path])
                    for result in results:
                        if 'error' in result:
                            print(f"Error: {result['error']}")
                        else:
                            print(f"Added: {result['file_name']} ({result['chunk_count']} chunks)")
                continue
            
            if user_input.lower() == 'stats':
                stats = researcher.get_knowledge_base_stats()
                print(f"\nKnowledge Base Statistics:")
                print(f"Documents: {stats.get('total_documents', 0)}")
                print(f"Chunks: {stats.get('total_chunks', 0)}")
                print(f"Queries: {stats.get('total_queries', 0)}")
                continue
            
            if user_input.lower() == 'list':
                docs = researcher.list_documents()
                print(f"\nDocuments ({len(docs)}):")
                for doc in docs:
                    print(f"  {doc['id']}: {doc['file_name']} ({doc['chunk_count']} chunks)")
                continue
            
            if user_input.startswith('search '):
                search_query = user_input[7:].strip()
                if search_query:
                    docs = researcher.search_documents(search_query)
                    print(f"\nFound {len(docs)} documents:")
                    for doc in docs:
                        print(f"  {doc['id']}: {doc['file_name']}")
                continue
            
            if user_input.startswith('report '):
                topic = user_input[7:].strip()
                if topic:
                    print(f"Generating research report on '{topic}'...")
                    result = researcher.generate_report(topic)
                    if 'error' in result:
                        print(f"Error: {result['error']}")
                    else:
                        print(f"Report generated successfully!")
                        if 'filename' in result:
                            print(f"Saved to: {result['filename']}")
                continue
            
            # Treat as a regular query
            print("Thinking...")
            result = researcher.query(user_input)
            
            print(f"\nAssistant: {result['response']}")
            
            if result['results']:
                print(f"\nFound {result['total_results']} relevant sources.")
                
                # Suggest follow-up questions
                suggestions = researcher.suggest_follow_up_questions(
                    user_input, result['results']
                )
                if suggestions:
                    print("\nSuggested follow-up questions:")
                    for suggestion in suggestions[:3]:
                        print(f"  • {suggestion}")
            
            print()
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
