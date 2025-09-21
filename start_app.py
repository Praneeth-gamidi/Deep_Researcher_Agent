"""
Simple startup script for Deep Researcher Agent with proper error handling.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import sentence_transformers
        import faiss
        import numpy
        import pandas
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def main():
    """Main startup function."""
    print("🔍 Deep Researcher Agent")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print("\nChoose your preferred mode:")
    print("1. Web Interface (Streamlit) - Recommended")
    print("2. Interactive CLI")
    print("3. Command Line Interface")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == "1":
                print("\n🚀 Starting Streamlit web interface...")
                print("The web interface will open at http://localhost:8501")
                print("Press Ctrl+C to stop the server")
                print()
                
                try:
                    # Set environment variables for better performance
                    env = os.environ.copy()
                    env['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
                    
                    subprocess.run([
                        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
                        "--server.headless", "false",
                        "--server.port", "8501",
                        "--browser.gatherUsageStats", "false"
                    ], env=env)
                except KeyboardInterrupt:
                    print("\n🛑 Server stopped by user")
                except FileNotFoundError:
                    print("❌ Streamlit not found. Installing...")
                    subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
                    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
                break
                
            elif choice == "2":
                print("\n🚀 Starting Interactive CLI...")
                subprocess.run([sys.executable, "main.py", "--mode", "interactive"])
                break
                
            elif choice == "3":
                print("\n🚀 Starting Command Line Interface...")
                subprocess.run([sys.executable, "main.py", "--mode", "cli"])
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break

if __name__ == "__main__":
    main()
