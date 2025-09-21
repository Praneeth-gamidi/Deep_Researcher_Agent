"""
Simple startup script for Deep Researcher Agent.
This script can be used for localhost and CodeMate Education platform deployment.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main startup function."""
    print("🔍 Deep Researcher Agent")
    print("=" * 50)
    print("Choose your preferred mode:")
    print("1. Web Interface (Streamlit) - Recommended")
    print("2. Interactive CLI")
    print("3. Command Line Interface")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == "1":
                print("\nStarting Streamlit web interface...")
                print("The web interface will open at http://localhost:8501")
                print("Press Ctrl+C to stop the server")
                print()
                
                try:
                    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
                except KeyboardInterrupt:
                    print("\nServer stopped.")
                except FileNotFoundError:
                    print("Streamlit not found. Installing...")
                    subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
                    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
                break
                
            elif choice == "2":
                print("\nStarting Interactive CLI...")
                subprocess.run([sys.executable, "main.py", "--mode", "interactive"])
                break
                
            elif choice == "3":
                print("\nStarting Command Line Interface...")
                subprocess.run([sys.executable, "main.py", "--mode", "cli"])
                break
                
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    main()
